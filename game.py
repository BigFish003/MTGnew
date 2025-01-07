import concurrent
import json
import os
import random
import re
import subprocess  # Not used, but was in your original import
from collections import Counter
import concurrent.futures
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DraftEnv(gym.Env):
    """
    A full 8-player draft environment with pass-to-the-right each round.
    - 3 rounds total.
    - In each round, all 8 players open a fresh 15-card pack (so 8 packs per round).
    - For 15 picks, each player:
       1) Picks 1 card from their current pack
       2) Passes the remainder to the right.
    - After 15 picks, those packs are empty. Move on to the next round.
    - After 3 rounds × 15 picks = 45 picks, you (seat 0) have 45 cards total.

    Observations (one-hot):
      shape = (60, num_cards_total)
        - Rows [0..44]: which cards you've picked so far
        - Rows [45..59]: your current 15-card pack
      If obs[i, c] = 1, row i is either:
        * 0..44 => your i-th pick is card c
        * 45..59 => your current pack slot (i-45) holds card c

    Actions:
      - pick a slot in [0..14] from your current 15-card pack
      - if that slot is empty/invalid, reward = -1, environment does NOT advance

    After the draft is over, you can call build_deck() to create a 60-card deck from:
      - 33 picks + 27 lands based on color distribution.
    """

    def __init__(self, json_path="fdn_cards.json"):
        super().__init__()

        # 1) Load card data
        with open(json_path, "r", encoding="utf-8") as f:
            self.all_fdn_cards = json.load(f)

        # 2) Build maps + color identity
        (self.card_id_map,
         self.id_card_map,
         self.rarity_pools,
         self.card_color_identity) = self._build_maps(self.all_fdn_cards)

        self.num_cards_total = len(self.card_id_map)

        # Draft structure
        self.num_players = 8        # 8 total seats
        self.num_packs = 3         # 3 rounds
        self.cards_per_pack = 15   # 15 picks per round

        # Action space: pick a slot [0..14]
        self.action_space = spaces.Discrete(self.cards_per_pack)

        # Observation space: (60 × num_cards_total)
        #   - 45 rows for your picks
        #   - 15 rows for your current pack
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(60, self.num_cards_total),
            dtype=np.float32
        )

        # We'll track your 45 picks here
        self.user_picks = []

        # For building the final deck
        self.basic_land_ids = {'W': None, 'U': None, 'B': None, 'R': None, 'G': None}

        # Internal state
        self.round_idx = 0    # which of the 3 rounds we are on
        self.pick_idx = 0     # which pick in the current round [0..14]
        self.done_drafting = False

    def reset(self, seed=None, options=None):
        """
        Start a new 3-round draft with 8 players. Each round, 8 new packs.
        Everyone passes to the right for 15 picks, then next round, etc.
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.round_idx = 0
        self.pick_idx = 0
        self.done_drafting = False
        self.user_picks = []

        # Build the 3 × 8 packs
        self.all_packs_by_round = []
        for _ in range(self.num_packs):
            round_packs = [self.make_pack() for __ in range(self.num_players)]
            self.all_packs_by_round.append(round_packs)

        # pack_holder[seat] = which pack index that seat is currently holding
        # Initially seat i holds pack i
        self.pack_holder = list(range(self.num_players))

        # Identify the basic land IDs, if we want to use them in build_deck later
        self._populate_basic_land_ids()

        return self._build_observation(), {}

    def step(self, action):
        print("stepped")
        """
        1) You (seat=0) pick from your current pack at slot `action`.
           - If invalid/empty, reward = -1, do NOT advance.
        2) Other 7 seats each pick randomly from their own pack.
        3) Pass packs to the right (seat i → seat i+1 mod 8).
        4) If we've done 15 picks in this round, move on to next round (if any).
        5) If we've done 3 rounds, we're done (45 picks).
        """
        if self.done_drafting:
            # Already finished
            return self._build_observation(), 0.0, True, {}

        # Which pack does seat 0 hold?
        user_pack_index = self.pack_holder[0]
        user_pack = self.all_packs_by_round[self.round_idx][user_pack_index]

        # Validate action
        if action < 0 or action >= self.cards_per_pack:
            return self._build_observation(), -1.0, False, {}
        if user_pack[action] is None:
            # User tried to pick from empty slot
            return self._build_observation(), -1.0, False, {}

        # 1) User picks card
        card_id = user_pack[action]
        self.user_picks.append(card_id)
        user_pack[action] = None

        # 2) Other seats pick randomly from their pack
        for seat in range(1, self.num_players):
            pack_index = self.pack_holder[seat]
            pack_cards = self.all_packs_by_round[self.round_idx][pack_index]
            valid_slots = [i for i, c in enumerate(pack_cards) if c is not None]
            if valid_slots:
                chosen_idx = random.choice(valid_slots)
                pack_cards[chosen_idx] = None

        # 3) Pass all packs to the right
        new_holder = [None] * self.num_players
        for seat in range(self.num_players):
            old_pack_index = self.pack_holder[seat]
            pass_to = (seat + 1) % self.num_players
            new_holder[pass_to] = old_pack_index
        self.pack_holder = new_holder

        # 4) Advance pick index
        self.pick_idx += 1
        if self.pick_idx >= self.cards_per_pack:
            # finished this round
            self.round_idx += 1
            self.pick_idx = 0
            if self.round_idx < self.num_packs:
                # new round => seat i holds pack i
                self.pack_holder = list(range(self.num_players))

        # 5) Check if we've done all 3 rounds
        if self.round_idx >= self.num_packs:
            self.done_drafting = True
            done = True
            deck = self.build_deck()
            results = self.test_in_parralell(deck)
            wins = 0
            length = len(results)
            for i in range(length):
                if results[i][1] == 1:
                    wins += 1
            reward = wins/length
            print(reward)
            return self._build_observation(), reward, done,False, {}
        else:
            done = False

        return self._build_observation(), 0.0, done,False, {}

    def make_pack(self):
        """
        Generate a 15-card pack (list of numeric IDs).
          - 1  = Basic Land (if available)
          - 10 = Commons
          - 3  = Uncommons
          - 1  = Rare or Mythic
        """
        pack_ids = []

        # 1 basic land
        if self.rarity_pools["basic_land"]:
            pack_ids.append(random.choice(self.rarity_pools["basic_land"]))

        # 10 commons
        if len(self.rarity_pools["common"]) >= 10:
            pack_ids.extend(random.sample(self.rarity_pools["common"], 10))
        else:
            pack_ids.extend(self.rarity_pools["common"])

        # 3 uncommons
        if len(self.rarity_pools["uncommon"]) >= 3:
            pack_ids.extend(random.sample(self.rarity_pools["uncommon"], 3))
        else:
            pack_ids.extend(self.rarity_pools["uncommon"])

        # 1 rare/mythic
        rare_or_mythic = self.rarity_pools["rare"] + self.rarity_pools["mythic"]
        if rare_or_mythic:
            pack_ids.append(random.choice(rare_or_mythic))

        # Ensure length = 15
        pack_ids = pack_ids[:self.cards_per_pack]
        while len(pack_ids) < self.cards_per_pack:
            pack_ids.append(None)
        return pack_ids

    def _build_maps(self, cards_data):
        """
        Assign each card a numeric ID, build:
          - rarity_pools: {common, uncommon, rare, mythic, basic_land} -> [IDs]
          - card_color_identity[card_id] = [...]
        """
        card_id_map = {}
        id_card_map = {}
        card_color_identity = {}
        rarity_pools = {
            "common": [],
            "uncommon": [],
            "rare": [],
            "mythic": [],
            "basic_land": []
        }
        idx = 0
        for card in cards_data:
            card_name = card["name"]
            if card_name not in card_id_map:
                card_id_map[card_name] = idx
                id_card_map[idx] = card_name
                idx += 1

            cid = card_id_map[card_name]
            # track color
            c_identity = card.get("color_identity", [])
            card_color_identity[cid] = c_identity

            # put into correct rarity bucket
            if card.get("is_basic_land", False):
                rarity_pools["basic_land"].append(cid)
            else:
                rar = card.get("rarity", "").lower()
                if rar in rarity_pools:
                    rarity_pools[rar].append(cid)
                else:
                    pass
        return card_id_map, id_card_map, rarity_pools, card_color_identity

    def _build_observation(self):
        """
        Build a (60, num_cards_total) one-hot observation:
          Rows [0..44]: your picks so far (if row i, col c = 1 => i-th pick is card c)
          Rows [45..59]: your current pack's 15 slots
        """
        obs = np.zeros((60, self.num_cards_total), dtype=np.float32)

        # Fill user picks into rows [0..len(self.user_picks)-1]
        for i, card_id in enumerate(self.user_picks):
            if i < 45:
                obs[i, card_id] = 1.0

        # If we're done drafting, no current pack to show
        if self.done_drafting:
            return obs

        # Build user "current pack" from the seat 0 perspective
        user_pack_index = self.pack_holder[0]
        user_pack = self.all_packs_by_round[self.round_idx][user_pack_index]

        # Place each slot in rows [45..59]
        for slot_i, cid in enumerate(user_pack):
            if cid is not None and (45 + slot_i) < 60:
                obs[45 + slot_i, cid] = 1.0

        return obs

    def get_mask(self):
        """
        Return a 15-element boolean mask of which slots are currently valid in your pack.
        """
        if self.done_drafting:
            # No pack
            return [False]*15
        user_pack_index = self.pack_holder[0]
        user_pack = self.all_packs_by_round[self.round_idx][user_pack_index]
        return [c is not None for c in user_pack]

    def build_deck(self, deck_size=60, main_count=33, land_count=27):
        """
        After drafting, build a 60-card deck from:
          - 33 picks from your user_picks
          - 27 basic lands chosen by color distribution
        """
        main_deck = self.user_picks[:main_count]

        # Count colors
        color_counts = {'W':0, 'U':0, 'B':0, 'R':0, 'G':0}
        for cid in main_deck:
            colors = self.card_color_identity.get(cid, [])
            for c in colors:
                if c in color_counts:
                    color_counts[c] += 1
        total_colors = sum(color_counts.values())

        # If no colors, all artifact? => just add 27 Islands
        if total_colors == 0:
            land_ids = [self.basic_land_ids['U']] * land_count
            return main_deck + land_ids

        # otherwise, allocate lands proportionally
        land_ids = []
        used = 0
        for c_symbol, c_count in color_counts.items():
            fraction = c_count / total_colors
            portion = int(round(fraction * land_count))
            if self.basic_land_ids[c_symbol] is not None:
                land_ids.extend([self.basic_land_ids[c_symbol]] * portion)
                used += portion

        # fix rounding mismatch
        diff = land_count - used
        if diff > 0:
            # add the color w/ largest fraction or default to 'U'
            top_color = max(color_counts, key=color_counts.get)
            top_land_id = self.basic_land_ids[top_color] or self.basic_land_ids['U']
            land_ids.extend([top_land_id] * diff)

        return main_deck + land_ids

    def _populate_basic_land_ids(self):
        """
        Looks in self.rarity_pools["basic_land"] to find numeric IDs for Plains, Island, Swamp, Mountain, Forest,
        by checking color_identity == ["W"] or ["U"] or ...
        """
        for land_id in self.rarity_pools["basic_land"]:
            ci = self.card_color_identity[land_id]
            if ci == ["W"]:
                self.basic_land_ids['W'] = land_id
            elif ci == ["U"]:
                self.basic_land_ids['U'] = land_id
            elif ci == ["B"]:
                self.basic_land_ids['B'] = land_id
            elif ci == ["R"]:
                self.basic_land_ids['R'] = land_id
            elif ci == ["G"]:
                self.basic_land_ids['G'] = land_id

    def test_deck(self, d1, d2=None):
        # 1) Convert the 'd1' list of IDs (or names) into a .dck file
        # 2) If d2 is not given, pick a random deck from a known set of .dck files

        # 3) Launch the Forge sim using the .dck for d1 and presumably a .dck for d2:
        command = [
            "java",
            "-jar",
            "forge-gui-desktop-2.0.01-jar-with-dependencies.jar",
            "sim",
            "-d",
            d1,  # Our newly created .dck file
            d2,  # Another .dck deck
            "-n",
            "1"
        ]
        working_directory = r"C:\mtgforge"
        process = subprocess.run(
            command,
            cwd=working_directory,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        x = process.stdout.split("\n")[-3]
        time = int(re.search(r"(\d+)\s*ms", x).group(1))

        # 5) Return the winner ID + time
        if "ms. Ai(2)" in x:
            return 2, time
        if "ms. Ai(1)" in x:
            return 1, time

    def test_in_parralell(self, d1):
        self._save_deck_to_dck_file(d1, deck_name="draft_deck")
        d1 = "draft_deck.dck"
        decks = ["DM_mdr.dck", "adventurer.dck"]
        results = []
        numper = 2
        executor_cls = concurrent.futures.ProcessPoolExecutor
        for i in range(int(len(decks)/numper)):
            with executor_cls() as executor:
                # Submit one job per deck
                futures = {
                    executor.submit(self.test_deck, d1, d2): d2
                    for d2 in decks
                }
                # Gather results as they complete
                for future in concurrent.futures.as_completed(futures):
                    d2 = futures[future]
                    winner, duration = future.result()
                    results.append((d2, winner, duration))
                    print("game played, lasted: " + str(duration))

        return results

    def _save_deck_to_dck_file(self, deck_ids, deck_name="draft_deck"):
        """
        Given a list of card IDs, writes a Forge .dck file to:
          C:\\Users\\samth\\AppData\\Roaming\\Forge\\decks\\constructed\\<deck_name>.dck
        using the format:
          [metadata]
          Name=<deck_name>
          [Main]
          3 Evolving Wilds|AFR|1
          2 Island|AFR|1
          ...
        """
        # Build the output path
        base_dir = r"C:\\Users\\samth\\AppData\\Roaming\\Forge\\decks\\constructed"
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"{deck_name}.dck")

        # Count duplicates, so if we have 4 copies of the same card, we write "4 CardName|AFR|1"
        deck_counter = Counter(deck_ids)

        # Start building lines for the .dck file
        lines = []
        lines.append("[metadata]")
        lines.append(f"Name={deck_name}")
        lines.append("[Main]")

        # Sort cards alphabetically by their names
        sorted_deck = sorted(deck_counter.items(), key=lambda item: self.id_card_map[item[0]])

        for cid, count in sorted_deck:
            card_name = self.id_card_map[cid]  # e.g. "Arcane Investigator"
            # If you have per-card set codes, you could retrieve them here:
            #   card_info = self.all_fdn_cards[...] or a dictionary mapping ID -> set_code
            #   set_code = card_info.get("set", "AFR")
            # For now, default to "AFR":
            set_code = "FDN"
            # Collector number is often "1" if we don't track the actual #.
            # We'll keep it simple:
            collector_num = "1"

            # Format: "4 Evolving Wilds|AFR|1"
            line = f"{count} {card_name}|{set_code}|{collector_num}"
            lines.append(line)
        lines.append("[Sideboard]")
        lines.append("")
        lines.append("[Avatar]")
        lines.append("")
        lines.append("[Planes]")
        lines.append("")
        lines.append("[Schemes]")
        lines.append("")
        lines.append("[Conspiracy]")
        lines.append("")
        lines.append("[Dungeon]")
        lines.append("")
        lines.append("[Attractions]")
        lines.append("")
        lines.append("[Contraptions]")
        lines.append("")

        # Write it out
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        return file_path