"""``print(report)`` reads as a list, folded where messages differ only in numbers."""
from kika.nuclear_data.model import ConversionReport


def test_an_empty_report_prints_clean():
    assert str(ConversionReport()) == "clean"


def test_kinds_are_grouped_in_order_under_their_counts():
    report = ConversionReport()
    report.warn("odd")
    report.unsupportedNode("refused")
    report.lost("gone")
    report.approximated("changed")
    assert str(report) == ("Losses (1)\n  - gone\n"
                           "Approximations (1)\n  - changed\n"
                           "Unsupported (1)\n  - refused\n"
                           "Warnings (1)\n  - odd")


def test_messages_that_differ_in_their_numbers_fold_into_one_line_with_ranges():
    report = ConversionReport()
    for mt in [2, *range(51, 90)]:
        report.warn(f"'MT{mt}' product 'n': no multiplicity; written 1, the number MT{mt} emits")
    assert str(report).splitlines()[1] == (
        "  - 'MT{2, 51-89}' product 'n': no multiplicity; written 1, the number "
        "MT{2, 51-89} emits (×40)")


def test_messages_that_differ_in_a_word_are_not_folded():
    report = ConversionReport()
    for pid, mts in (("n", range(51, 53)), ("H1", range(600, 603))):
        for mt in mts:
            report.warn(f"'MT{mt}' product '{pid}': no multiplicity")
    assert str(report).splitlines()[1:] == [
        "  - 'MT51-52' product 'n': no multiplicity (×2)",
        "  - 'MT600-602' product 'H1': no multiplicity (×3)"]


def test_decimals_and_exact_repeats_fold():
    report = ConversionReport()
    for l, j in ((0, 0.5), (1, 0.5), (1, 1.5)):
        report.lost(f"URR L={l} J={j}: not serialized")
    report.warn("same")
    report.warn("same")
    assert str(report).splitlines() == [
        "Losses (3)",
        "  - URR L=0-1 J=0.5: not serialized (×2)",
        "  - URR L=1 J=1.5: not serialized",
        "Warnings (2)",
        "  - same (×2)"]


def test_a_digit_inside_a_name_is_not_folded_with_the_number_beside_it():
    report = ConversionReport()
    for mt, pid in ((600, "H1"), (601, "H1"), (650, "H2"), (651, "H2"), (51, "U238_e1"),
                    (52, "U238_e2")):
        report.warn(f"'MT{mt}' product '{pid}': no multiplicity")
    assert str(report).splitlines()[1:] == [
        "  - 'MT600-601' product 'H1': no multiplicity (×2)",
        "  - 'MT650-651' product 'H2': no multiplicity (×2)",
        "  - 'MT51-52' product 'U238_e1-2': no multiplicity (×2)"]


def test_a_long_scatter_is_elided_and_the_lists_are_untouched():
    report = ConversionReport()
    for mt in range(0, 40, 2):
        report.warn(f"MT{mt} odd")
    assert "MT{0, 2, 4, …, 36, 38} odd (×20)" in str(report)
    assert len(report.warnings) == 20
    assert repr(report) == "ConversionReport(20 warnings)"
