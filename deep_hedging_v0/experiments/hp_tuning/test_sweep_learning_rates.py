import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.hp_tuning import sweep_learning_rates as sweep
from experiments.hp_tuning import run_experiment as workflow


class LearningRateSweepTests(unittest.TestCase):
    def test_all_experiments_have_twelve_unique_configurations(self):
        scripts = sorted(sweep.SIMULATIONS_DIR.glob("*.py"))
        self.assertEqual(len(scripts), 24)
        for script in scripts:
            with self.subTest(script=script.name):
                rates = sweep.trial_rates(script)
                self.assertEqual(len(rates), 12)
                self.assertEqual(len(set(rates)), 12)
                actor_default = 5e-6 if script.stem.endswith("_rnn") else 5e-5
                inactive = script.stem.startswith("inactive_")
                self.assertIn((actor_default, 1e-3 if inactive else None), rates)
                if inactive:
                    self.assertEqual(len({actor for actor, _ in rates}), 3)
                    self.assertEqual(len({inactor for _, inactor in rates}), 4)
                else:
                    self.assertTrue(all(inactor is None for _, inactor in rates))

    def test_command_covers_cli_and_config_only_scripts(self):
        for script in sweep.SIMULATIONS_DIR.glob("*.py"):
            with self.subTest(script=script.name):
                command = sweep.build_trial_command(script, 2.5e-5, 7)
                self.assertEqual(command[:4], [sweep.sys.executable, str(script), "--seed", "7"])
                if "--actor_learning_rate" in script.read_text(encoding="utf-8"):
                    self.assertEqual(command[4:], ["--actor_learning_rate", "2.5e-05"])
                else:
                    self.assertEqual(len(command), 4)

    def test_rates_and_budgets_override_inherited_environment(self):
        with patch.dict("os.environ", {"DEEP_HEDGING_JOINT_LEARNING_RATE": "999"}):
            environment = sweep.trial_environment(2.5e-5, 3e-4, 60, 40, 20, 400)
        self.assertEqual(float(environment["DEEP_HEDGING_LEARNING_RATE"]), 2.5e-5)
        self.assertEqual(float(environment["DEEP_HEDGING_JOINT_LEARNING_RATE"]), 2.5e-5)
        self.assertEqual(float(environment["DEEP_HEDGING_INACTOR_LEARNING_RATE"]), 3e-4)
        self.assertEqual(environment["DEEP_HEDGING_NUM_EPOCHS"], "60")
        self.assertEqual(environment["DEEP_HEDGING_POLICY_EPOCHS"], "40")
        self.assertEqual(environment["DEEP_HEDGING_INACTION_EPOCHS"], "20")
        self.assertEqual(environment["DEEP_HEDGING_NUM_EPISODES"], "400")
        self.assertEqual(environment["WANDB_MODE"], "disabled")
        self.assertEqual(environment["DEEP_HEDGING_SAVE_ARTIFACTS"], "0")

    def test_trial_parses_metrics_and_preserves_logs(self):
        arguments = argparse.Namespace(epochs=60, policy_epochs=40, inaction_epochs=20, episodes=400, seed=7)
        script = sweep.SIMULATIONS_DIR / "inactive_bsm_hedge_call_mlp.py"
        outputs = (
            ("SWEEP_RESULT cvar_95=99\nSWEEP_RESULT cvar_95=-1.25e-03\n", -0.00125),
            ("SWEEP_RESULT cvar_95=nan\n", None),
            ("SWEEP_RESULT cvar_95=inf\n", None),
            ("SWEEP_RESULT cvar_95=invalid\n", None),
            ("no result\n", None),
        )
        with tempfile.TemporaryDirectory() as directory:
            for stdout, expected in outputs:
                with self.subTest(stdout=stdout), patch.object(
                    sweep.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout),
                ) as run:
                    result = sweep.run_trial(script, 2.5e-5, 3e-4, arguments, 1, Path(directory))
                    self.assertEqual(result["cvar_95"], expected)
                    self.assertEqual(result["joint_learning_rate"], 2.5e-5)
                    self.assertEqual(Path(result["log_path"]).read_text(), stdout)
                    self.assertEqual(run.call_args.kwargs["env"]["DEEP_HEDGING_INACTOR_LEARNING_RATE"], "0.0003")

    def test_dry_run_has_288_trials_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            capture = io.StringIO()
            with patch("sys.argv", ["sweep", "--dry-run", "--output", str(output)]), \
                    redirect_stdout(capture), patch.object(sweep, "run_trial") as run:
                sweep.main()
            self.assertEqual(capture.getvalue().count("trial="), 288)
            self.assertEqual(list(Path(directory).iterdir()), [])
            run.assert_not_called()

    def test_csv_and_comparison_exclude_failed_trials(self):
        def fake_trial(script, actor_rate, inactor_rate, arguments, trial, log_dir):
            inactive = script.stem.startswith("inactive_")
            return {
                "experiment": script.stem, "trial": trial,
                "actor_learning_rate": actor_rate, "inactor_learning_rate": inactor_rate,
                "cvar_95": -999 if trial == 1 else (2.0 if inactive else 3.0),
                "exit_code": 1 if trial == 1 else 0,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            argv = ["sweep", "--output", str(output), "--scripts",
                    "bsm_hedge_call_mlp.py", "inactive_bsm_hedge_call_mlp.py",
                    "bsm_hedge_call_mlp.py"]
            capture = io.StringIO()
            with patch("sys.argv", argv), redirect_stdout(capture), \
                    patch.object(sweep, "run_trial", side_effect=fake_trial) as run:
                sweep.main()
            self.assertEqual(run.call_count, 24)
            with output.open(newline="") as result_file:
                rows = list(csv.DictReader(result_file))
            self.assertEqual(len(rows), 24)
            self.assertEqual(set(rows[0]), set(sweep.RESULT_FIELDS))
            self.assertIn("(11/12 valid trials)", capture.getvalue())
            self.assertIn("active-minus-inactive CVaR=1.0000; inactive wins", capture.getvalue())
            best_dir = output.with_suffix(".best")
            for name, expected_cvar in (("bsm_hedge_call_mlp", 3.0), ("inactive_bsm_hedge_call_mlp", 2.0)):
                config = json.loads((best_dir / f"{name}.json").read_text())
                self.assertEqual(config["experiment"], name)
                self.assertEqual(config["selected_trial"], 2)
                self.assertEqual(config["cvar_95"], expected_cvar)
                self.assertEqual(config["valid_trials"], 11)
                self.assertEqual(config["attempted_trials"], 12)
                self.assertEqual(config["environment"]["DEEP_HEDGING_NUM_EPOCHS"], "60")
                launcher = (best_dir / f"{name}.slurm").read_text()
                self.assertIn("export DEEP_HEDGING_SAVE_ARTIFACTS=1", launcher)
                self.assertIn('--seed "${RUN_SEED:-0}"', launcher)
                self.assertIn(f"--actor_learning_rate {config['actor_learning_rate']}", launcher)
                self.assertIn(f"export DEEP_HEDGING_JOINT_LEARNING_RATE={config['actor_learning_rate']}", launcher)
                self.assertIn("export DEEP_HEDGING_INACTOR_LEARNING_RATE=" + config["environment"]["DEEP_HEDGING_INACTOR_LEARNING_RATE"], launcher)
                syntax = subprocess.run(["bash", "-n", str(best_dir / f"{name}.slurm")], capture_output=True, text=True)
                self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_saved_launchers_cover_all_experiment_variants(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            arguments = argparse.Namespace(
                output=Path(directory) / "results.csv", epochs=60,
                policy_epochs=40, inaction_epochs=20, episodes=400, seed=7,
            )
            for script in sweep.SIMULATIONS_DIR.glob("*.py"):
                with self.subTest(script=script.name):
                    actor_rate, inactor_rate = sweep.trial_rates(script)[2]
                    best = {"actor_learning_rate": actor_rate, "inactor_learning_rate": inactor_rate,
                            "cvar_95": 1.0, "trial": 3}
                    with patch.dict("os.environ", {"DEEP_HEDGING_NUM_PATHS": "75"}):
                        path = sweep.save_best_config(script, best, arguments, 12)
                    config = json.loads(path.read_text())
                    self.assertEqual(config["actor_learning_rate"], actor_rate)
                    self.assertEqual(config["inactor_learning_rate"], inactor_rate)
                    self.assertEqual(config["seed"], 7)
                    self.assertEqual(config["environment"]["DEEP_HEDGING_NUM_PATHS"], "75")
                    launcher = path.with_suffix(".slurm").read_text()
                    self.assertIn(f"export DEEP_HEDGING_LEARNING_RATE={actor_rate}", launcher)
                    self.assertIn(f"export DEEP_HEDGING_JOINT_LEARNING_RATE={actor_rate}", launcher)
                    self.assertIn(f"export DEEP_HEDGING_INACTOR_LEARNING_RATE={inactor_rate if inactor_rate is not None else 1e-3}", launcher)
                    self.assertIn('--seed "${RUN_SEED:-7}"', launcher)
                    self.assertEqual("--actor_learning_rate" in launcher, "--actor_learning_rate" in script.read_text())
                    syntax = subprocess.run(["bash", "-n", str(path.with_suffix(".slurm"))], capture_output=True, text=True)
                    self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_completed_experiment_is_saved_before_next_experiment(self):
        def fake_trial(script, actor_rate, inactor_rate, arguments, trial, log_dir):
            if script.stem.startswith("inactive_"):
                raise KeyboardInterrupt
            return {"experiment": script.stem, "trial": trial,
                    "actor_learning_rate": actor_rate, "inactor_learning_rate": inactor_rate,
                    "cvar_95": float(trial), "exit_code": 0}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            argv = ["sweep", "--output", str(output), "--scripts",
                    "bsm_hedge_call_mlp.py", "inactive_bsm_hedge_call_mlp.py"]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()), \
                    patch.object(sweep, "run_trial", side_effect=fake_trial), self.assertRaises(KeyboardInterrupt):
                sweep.main()
            best_dir = output.with_suffix(".best")
            config = json.loads((best_dir / "bsm_hedge_call_mlp.json").read_text())
            self.assertEqual(config["selected_trial"], 1)
            self.assertEqual(config["valid_trials"], 12)
            self.assertFalse((best_dir / "inactive_bsm_hedge_call_mlp.json").exists())

    def test_all_failed_trials_do_not_export_a_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            argv = ["sweep", "--output", str(output), "--scripts", "bsm_hedge_call_mlp.py"]
            failed = {"experiment": "bsm_hedge_call_mlp", "cvar_95": -999, "exit_code": 1}
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()), \
                    patch.object(sweep, "run_trial", return_value=failed):
                sweep.main()
            self.assertFalse(output.with_suffix(".best").exists())

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            output.write_text("existing results", encoding="utf-8")
            with patch("sys.argv", ["sweep", "--output", str(output)]), \
                    redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), \
                    patch.object(sweep, "run_trial") as run, self.assertRaises(SystemExit):
                sweep.main()
            self.assertEqual(output.read_text(), "existing results")
            run.assert_not_called()

    def test_invalid_budget_is_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                sweep.positive_int(value)


class TuneThenTrainTests(unittest.TestCase):
    def test_real_child_inherits_module_search_path(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = self.arguments(directory)
            arguments.output.mkdir()
            module_dir = Path(directory) / "cluster_modules"
            module_dir.mkdir()
            (module_dir / "cluster_dependency_probe.py").write_text("VALUE = 1.25\n", encoding="utf-8")
            script = Path(directory) / "probe.py"
            script.write_text(
                "import os\nimport sys\nimport cluster_dependency_probe\n"
                "print('cwd=' + os.getcwd())\n"
                "print('python=' + sys.executable)\n"
                "print('project=' + os.environ['WANDB_PROJECT'])\n"
                "print('SWEEP_RESULT cvar_95=' + str(cluster_dependency_probe.VALUE))\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PYTHONPATH": str(module_dir), "WANDB_PROJECT": "original-project"}), \
                    redirect_stdout(io.StringIO()):
                result = workflow.run_training(script, 5e-5, None, 0, 36, arguments, "tuning", 1)
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["cvar_95"], 1.25)
            log = Path(result["log_path"]).read_text()
            self.assertIn("cwd=" + str(workflow.ROOT.parent), log)
            self.assertIn("python=" + sweep.sys.executable, log)
            self.assertIn("project=original-project", log)

    def test_unset_module_path_and_project_stay_unset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop("PYTHONPATH", None)
            os.environ.pop("WANDB_PROJECT", None)
            arguments = self.arguments(directory)
            arguments.output.mkdir()
            with patch.object(workflow.subprocess, "Popen") as popen, redirect_stdout(io.StringIO()):
                process = popen.return_value.__enter__.return_value
                process.stdout = io.StringIO("SWEEP_RESULT cvar_95=1\n")
                process.wait.return_value = 0
                workflow.run_training(sweep.SIMULATIONS_DIR / "bsm_hedge_call_mlp.py", 5e-5, None, 0, 36, arguments, "tuning", 1)
            environment = popen.call_args.kwargs["env"]
            self.assertNotIn("PYTHONPATH", environment)
            self.assertNotIn("WANDB_PROJECT", environment)

    def arguments(self, directory):
        return argparse.Namespace(
            output=Path(directory) / "experiment_job123", epochs=60,
            policy_epochs=40, inaction_epochs=20, episodes=400,
        )

    def fake_result(self, script, actor_rate, inactor_rate, seed, epochs, arguments, phase, index):
        if phase == "final":
            self.assertTrue((arguments.output / "best_config.json").exists())
        return {
            "phase": phase, "index": index, "seed": seed,
            "actor_learning_rate": actor_rate, "inactor_learning_rate": inactor_rate,
            "epochs": epochs, "cvar_95": float(abs(index - 5)), "exit_code": 0,
        }

    def test_all_experiments_tune_then_run_ten_full_seeds(self):
        for script in sweep.SIMULATIONS_DIR.glob("*.py"):
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as directory:
                arguments = self.arguments(directory)
                with patch.object(workflow, "run_training", side_effect=self.fake_result) as run, \
                        redirect_stdout(io.StringIO()):
                    workflow.run_experiment(script, arguments)
                calls = [call.args for call in run.call_args_list]
                self.assertEqual(len(calls), 22)
                expected_epochs = 60 if script.stem.startswith("inactive_") else 36
                for call in calls[:12]:
                    self.assertEqual(call[3], 0)
                    self.assertEqual(call[4], expected_epochs)
                    self.assertEqual(call[6], "tuning")
                self.assertEqual([call[3] for call in calls[12:]], list(range(1, 11)))
                for call in calls[12:]:
                    self.assertEqual(call[1:3], calls[4][1:3])
                    self.assertEqual(call[4], 60)
                    self.assertEqual(call[6], "final")
                config = json.loads((arguments.output / "best_config.json").read_text())
                self.assertEqual(config["selected_trial"], 5)
                self.assertEqual(config["final_epochs"], 60)
                self.assertEqual(config["tuning_epochs"], expected_epochs)
                self.assertEqual(config["policy_epochs"], 40)
                self.assertEqual(config["inaction_epochs"], 20)
                with (arguments.output / "runs.csv").open() as output:
                    self.assertEqual(len(list(csv.DictReader(output))), 22)

    def test_failed_candidate_cannot_win_and_final_failure_is_reported(self):
        def failed_run(*args):
            result = self.fake_result(*args)
            if args[7] == 5:
                result.update(cvar_95=-999, exit_code=1)
            return result

        with tempfile.TemporaryDirectory() as directory:
            arguments = self.arguments(directory)
            with patch.object(workflow, "run_training", side_effect=failed_run) as run, \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "seeds \\[5\\]"):
                workflow.run_experiment(sweep.SIMULATIONS_DIR / "bsm_hedge_call_mlp.py", arguments)
            self.assertEqual(run.call_count, 22)
            config = json.loads((arguments.output / "best_config.json").read_text())
            self.assertEqual(config["selected_trial"], 4)
            self.assertEqual(config["valid_trials"], 11)

    def test_all_failed_tuning_prevents_final_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = self.arguments(directory)
            with patch.object(workflow, "run_training", return_value={"exit_code": 1, "cvar_95": None}) as run, \
                    self.assertRaisesRegex(RuntimeError, "All 12 tuning trials failed"):
                workflow.run_experiment(sweep.SIMULATIONS_DIR / "bsm_hedge_call_mlp.py", arguments)
            self.assertEqual(run.call_count, 12)
            self.assertFalse((arguments.output / "best_config.json").exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = self.arguments(directory)
            arguments.output.mkdir()
            with patch.object(workflow, "run_training") as run, self.assertRaises(FileExistsError):
                workflow.run_experiment(sweep.SIMULATIONS_DIR / "bsm_hedge_call_mlp.py", arguments)
            run.assert_not_called()

    def test_online_wandb_streaming_and_phase_settings(self):
        for phase, seed in (("tuning", 0), ("final", 3)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                arguments = self.arguments(directory)
                arguments.output.mkdir()
                script = sweep.SIMULATIONS_DIR / "inactive_bsm_hedge_call_mlp.py"
                with patch.object(workflow.subprocess, "Popen") as popen, \
                        patch.dict("os.environ", {"WANDB_MODE": "disabled", "WANDB_RUN_ID": "old-run", "WANDB_RESUME": "must"}), \
                        redirect_stdout(io.StringIO()) as capture:
                    process = popen.return_value.__enter__.return_value
                    process.stdout = io.StringIO("training progress\nSWEEP_RESULT cvar_95=1.25e-3\n")
                    process.wait.return_value = 0
                    result = workflow.run_training(script, 2.5e-5, 3e-4, seed, 60, arguments, phase, 1)
                environment = popen.call_args.kwargs["env"]
                self.assertEqual(environment["WANDB_MODE"], "online")
                self.assertEqual(environment["WANDB_JOB_TYPE"], phase)
                self.assertEqual(environment["WANDB_RUN_GROUP"], arguments.output.name)
                self.assertIn("inactor_lr=0.0003", environment["WANDB_TAGS"])
                self.assertNotIn("WANDB_RUN_ID", environment)
                self.assertNotIn("WANDB_RESUME", environment)
                self.assertEqual(environment["DEEP_HEDGING_SAVE_ARTIFACTS"], "1" if phase == "final" else "0")
                self.assertEqual(environment["DEEP_HEDGING_LEARNING_RATE"], "2.5e-05")
                self.assertEqual(environment["DEEP_HEDGING_JOINT_LEARNING_RATE"], "2.5e-05")
                self.assertEqual(environment["DEEP_HEDGING_INACTOR_LEARNING_RATE"], "0.0003")
                self.assertEqual(popen.call_args.args[0][2:4], ["--seed", str(seed)])
                self.assertIn("training progress", capture.getvalue())
                self.assertIn("training progress", Path(result["log_path"]).read_text())
                self.assertEqual(result["cvar_95"], 0.00125)


if __name__ == "__main__":
    unittest.main()