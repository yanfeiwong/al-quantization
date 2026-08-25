"""Build all currently implemented publication figures."""

from fig01_state_landscape import build as build_fig01
from fig02_al_encoding import build as build_fig02
from fig03_state_heterogeneity import build as build_fig03
from fig04_controlled_fidelity import build as build_fig04
from fig05_training_curves import build as build_fig05


def main() -> None:
    generated = {}
    generated.update({f"fig01_{k}": v for k, v in build_fig01().items()})
    generated.update({f"fig02_{k}": v for k, v in build_fig02().items()})
    generated.update({f"fig03_{k}": v for k, v in build_fig03().items()})
    generated.update({f"fig04_{k}": v for k, v in build_fig04().items()})
    generated.update({f"fig05_{k}": v for k, v in build_fig05().items()})
    for path in generated.values():
        print(path)


if __name__ == "__main__":
    main()
