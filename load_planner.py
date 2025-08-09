from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import csv
import xml.etree.ElementTree as ET


@dataclass
class ULD:
    """Unité de chargement (Unit Load Device)."""
    id: str
    length: float
    width: float
    height: float
    weight: float
    priority: int = 0


@dataclass
class CargoHold:
    """Représentation simplifiée d'une soute."""
    length: float
    width: float
    height: float


class LoadPlanner:
    """Planificateur de chargement utilisant une heuristique simple."""

    def plan_load(
        self, hold: CargoHold, ulds: List[ULD]
    ) -> Tuple[List[Dict[str, Any]], List[ULD]]:
        """Retourne les placements et les ULD non placés."""
        placements: List[Dict[str, Any]] = []
        unplaced: List[ULD] = []

        # Trier par priorité puis par poids décroissants
        ulds_sorted = sorted(ulds, key=lambda u: (-u.priority, -u.weight))

        x = y = z = 0.0
        row_width = 0.0
        layer_height = 0.0

        for uld in ulds_sorted:
            # Vérifier que l'ULD peut entrer dans la soute
            if (
                uld.length > hold.length
                or uld.width > hold.width
                or uld.height > hold.height
            ):
                unplaced.append(uld)
                continue

            if x + uld.length > hold.length:
                x = 0.0
                y += row_width
                row_width = 0.0

            if y + uld.width > hold.width:
                x = 0.0
                y = 0.0
                z += layer_height
                layer_height = 0.0

            if z + uld.height > hold.height:
                unplaced.append(uld)
                continue

            position = (x, y, z)
            placements.append(
                {
                    "uld_id": uld.id,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                    "order": len(placements) + 1,
                }
            )

            x += uld.length
            row_width = max(row_width, uld.width)
            layer_height = max(layer_height, uld.height)

        return placements, unplaced

    @staticmethod
    def export_plan_csv(placements: List[Dict[str, Any]], path: str) -> None:
        """Exporte le plan de chargement en CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["uld_id", "x", "y", "z", "order"])
            writer.writeheader()
            for p in placements:
                writer.writerow(p)

    @staticmethod
    def export_plan_xml(placements: List[Dict[str, Any]], path: str) -> None:
        """Exporte le plan de chargement en XML."""
        root = ET.Element("LoadPlan")
        for p in placements:
            uld_elem = ET.SubElement(root, "ULD", id=str(p["uld_id"]), order=str(p["order"]))
            ET.SubElement(
                uld_elem,
                "Position",
                x=str(p["x"]),
                y=str(p["y"]),
                z=str(p["z"]),
            )
        tree = ET.ElementTree(root)
        tree.write(path, encoding="utf-8", xml_declaration=True)
