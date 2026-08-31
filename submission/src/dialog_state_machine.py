"""Dialog slot lifecycle and phase management."""

from __future__ import annotations

from .text_utils import _normal, _terms


class DialogStateMachine:
    """Accumulate slots, retire superseded preferences, and expose dialog phase."""

    MATERIALS = {
        "acetate", "acrylic", "aluminum", "aluminium", "bamboo", "brass",
        "canvas", "carbon", "cashmere", "ceramic", "chiffon", "chrome",
        "corduroy", "cork", "cotton", "crystal", "denim", "elastic", "fabric",
        "faux fur", "faux leather", "felt", "fiberglass", "flannel", "foam",
        "fur", "glass", "gold", "hemp", "iron", "jute", "lace", "latex", "leather",
        "linen", "lycra", "mesh", "metal", "microfiber", "modal", "neoprene", "nickel",
        "nylon", "paper", "pearl", "pewter", "plastic", "pleather", "polyamide", "polyester",
        "polypropylene", "polyurethane", "porcelain", "rayon", "resin", "rubber", "satin",
        "silicone", "silk", "silver", "spandex", "stainless steel", "steel", "suede",
        "synthetic", "tencel", "textile", "titanium", "tungsten", "velvet", "vinyl",
        "viscose", "wood", "zinc",
    }
    COLORS = {
        "aqua", "aquamarine", "apricot", "azure", "beige", "black", "blush", "blue",
        "bronze", "brown", "burgundy", "camel", "charcoal", "chocolate", "clear",
        "cobalt", "copper", "coral", "cream", "cyan", "ecru", "emerald", "fluorescent",
        "fuchsia", "gold", "gray", "green", "grey", "indigo", "ivory", "khaki", "lavender", "lilac", "lime",
        "magenta", "maroon", "mint", "multicolor", "mustard", "navy", "olive", "orange",
        "neon", "peach", "periwinkle", "pink", "plum", "purple", "raspberry", "red", "rose",
        "rust", "salmon", "sand", "scarlet", "silver", "slate", "tan", "taupe", "teal", "turquoise", "violet",
        "white", "wine", "yellow",
    }

    @classmethod
    def slot_kind(cls, value: str, source: str) -> str:
        terms = set(_terms(value))
        lowered = value.lower()
        if source == "category":
            return "category"
        if "budget" in lowered or "$" in value or "under " in lowered:
            return "budget"
        if terms & cls.MATERIALS:
            return "material"
        if "color" in terms or terms & cls.COLORS:
            return "color"
        if terms & {"size", "sizing", "width", "wide", "narrow"}:
            return "size"
        if terms & {"hiking", "running", "gym", "winter", "outdoor", "work"}:
            return "use_case"
        if terms & {"style", "fit", "sleeve", "neck", "department"}:
            return "style"
        return "feature"

    @classmethod
    def apply(
        cls,
        state: dict,
        observations: list[tuple[str, str]],
        turn: int,
        override: bool,
    ) -> None:
        if override:
            # An override erases prior soft preferences, while retaining the
            # category and independently confirmed hard constraints.
            for slot in state["slots"]:
                if slot["active"] and slot["source"] == "preference":
                    slot["active"] = False

        active_values = {
            _normal(slot["value"]) for slot in state["slots"] if slot["active"]
        }
        for value, source in observations:
            normalized = _normal(value)
            if not normalized or normalized in active_values:
                continue
            kind = cls.slot_kind(value, source)
            if source == "override":
                # A later override rewrites an earlier override of the same
                # slot type but does not erase unrelated hard requirements.
                for slot in state["slots"]:
                    if (
                        slot["active"]
                        and slot["source"] == "override"
                        and slot["kind"] == kind
                    ):
                        slot["active"] = False
            if kind == "category":
                for slot in state["slots"]:
                    if slot["active"] and slot["kind"] == "category":
                        slot["active"] = False
            state["slots"].append({
                "kind": kind,
                "value": value,
                "source": source,
                "turn": turn,
                "active": True,
            })
            active_values.add(normalized)

        state["constraints"] = [
            slot["value"] for slot in state["slots"] if slot["active"]
        ]
        state["phase"] = "intent_override" if override else (
            "refinement" if len(state["constraints"]) > 1 else "discovery"
        )
