import numpy as np

class DraftRenderer:
    """
    A text-based renderer for the DraftEnv that emulates the 'draft' view:
      - The center shows the current pack (slots 0..14).
      - The right side lists all the picks made so far (up to 45).
    """
    def __init__(self, id_card_map):
        """
        id_card_map: dict[int, str] mapping card_id -> card_name
        """
        self.id_card_map = id_card_map

    def render(self, obs):
        """
        Renders the draft state given the observation from the environment.

        The obs is a (60 x num_cards_total) one-hot matrix:
          - rows [0..44]  => each pick the user made (max 45 picks)
          - rows [45..59] => the 15 card slots in the current pack
        """
        # 1) Parse user picks (rows 0..44)
        user_picks = []
        for pick_idx in range(45):
            row = obs[pick_idx]  # shape = (num_cards_total,)
            card_ids = np.where(row == 1.0)[0]
            if len(card_ids) > 0:
                # For a normal pick, there's typically exactly 1 ID active
                picked_id = card_ids[0]
                user_picks.append(self.id_card_map[picked_id])

        # 2) Parse current pack (rows 45..59)
        current_pack = []
        for slot_idx in range(15):
            row = obs[45 + slot_idx]  # shape = (num_cards_total,)
            card_ids = np.where(row == 1.0)[0]
            if len(card_ids) > 0:
                # There's normally at most 1 ID for a given slot
                card_id = card_ids[0]
                card_name = self.id_card_map[card_id]
                current_pack.append(card_name)
            else:
                # This slot is empty
                current_pack.append(None)

        # 3) Produce a text-based layout
        # We'll just print the current pack in one section
        # and the user's picks in another. In a real UI, you'd
        # place them side by side or in a typical "drafting" layout.
        self._render_text(user_picks, current_pack)

    def _render_text(self, user_picks, current_pack):
        """
        Utility function that prints a simple text-based layout.
        """
        # Header
        print("=" * 40)
        print("          MTG DRAFT VIEW          ")
        print("=" * 40)

        print("CURRENT PACK (slots 0..14):")
        for i, card_name in enumerate(current_pack):
            if card_name is None:
                print(f"  Slot {i}: [EMPTY]")
            else:
                print(f"  Slot {i}: {card_name}")

        print("")
        print("YOUR PICKS (so far):")
        if not user_picks:
            print("  No picks yet.")
        else:
            for i, pick_name in enumerate(user_picks):
                print(f"  Pick {i+1}: {pick_name}")

        print("=" * 40)
        print("")
