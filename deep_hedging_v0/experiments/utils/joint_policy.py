import torch
import torch.nn as nn

class JointPolicy(nn.Module):
    def __init__(self, actor_policy, inactor_policy, action_dim, device):
        super().__init__()
        self.actor_policy = actor_policy
        self.inactor_policy = inactor_policy
        self.action_dim = action_dim
        self.device = device
        self.prev_action = None

    def _ensure_prev_action(self, action_like: torch.Tensor):
        if (
            self.prev_action is None
            or self.prev_action.shape != action_like.shape
            or self.prev_action.device != action_like.device
            or self.prev_action.dtype  != action_like.dtype
        ):
            self.prev_action = torch.zeros_like(action_like)

    def forward(self, td):
        # don't mutate collector buffers
        td = td.clone(False)

        # run both policies
        td = self.actor_policy(td)      # writes "action"
        td = self.inactor_policy(td)    # writes "inact"

        action = td["action"]                    # shape: (*batch_shape, A)
        inact  = td["inact"]                     # could be (*batch_shape), (*batch_shape,1), or (*batch_shape,A)
        A = action.shape[-1]
        batch_shape = action.shape[:-1]
        batch_elems = action.numel() // A

        # keep original action for PPO update of the actor
        td = td.set("original_action", action.clone())

        # prev action buffer
        self._ensure_prev_action(action)

        # numeric mask
        mask = inact.to(action.dtype) if inact.dtype == torch.bool else inact

        # -------- robust mask shaping --------
        mnum = mask.numel()
        anum = action.numel()

        if mnum == batch_elems:
            # mask has no action dim -> expand to all action comps
            mask = mask.reshape(*batch_shape, 1).expand_as(action)
        elif mnum == batch_elems * 1:
            # explicit singleton action dim -> expand
            mask = mask.reshape(*batch_shape, 1).expand_as(action)
        elif mnum == anum:
            # already per-action -> just view to action's shape
            mask = mask.reshape_as(action)
        else:
            raise RuntimeError(
                f"Inaction mask size {mnum} is incompatible with action size {anum} "
                f"(batch_elems={batch_elems}, A={A}, batch_shape={batch_shape}). "
                f"Expected {batch_elems}, {batch_elems*1}, or {anum} elements."
            )
        # -------------------------------------

        # mix: mask==1 keeps previous action; mask==0 uses current action
        final_action = torch.where(mask > 0.5, self.prev_action, action)

        td = td.set("action", final_action)
        self.prev_action = final_action.detach()
        return td

    def reset(self):
        self.prev_action = None
