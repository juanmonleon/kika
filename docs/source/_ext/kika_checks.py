"""``kika-covariance-checks`` and ``kika-data-checks``: one group of ``CHECKS`` or one family of ``DATA_CHECKS``.

The texts are the ones the reports and the app show. Nodes are built directly: the texts hold
``|rho|``, a table in Markdown and a substitution in rST.
"""
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

_LEVELS = (("defect", "Defect"), ("warn", "Warning"), ("note", "Note"))
_LOG = logging.getLogger(__name__)


def _paragraph(*children):
    return nodes.paragraph("", "", *children)


def _covariance_tables():
    from kika.endf.checks.descriptions import CHECKS, GROUPS

    return CHECKS, GROUPS


def _data_tables():
    from kika.endf.checks.data.descriptions import DATA_CHECKS, DATA_FAMILIES

    return DATA_CHECKS, DATA_FAMILIES


class _Checks(Directive):
    required_arguments = 1
    kind = ""
    tables = None

    def run(self):
        table, groups = type(self).tables()
        group = self.arguments[0]
        if group not in groups:
            raise self.error(f"no {self.kind} check group {group!r}; the groups are {', '.join(groups)}")
        env = self.state.document.settings.env
        shown = getattr(env, "kika_check_groups", {})
        shown.setdefault(self.kind, set()).add(group)
        env.kika_check_groups = shown
        return [self._section(name, table[name].to_dict()) for name in groups[group]]

    def _section(self, name, d):
        anchor = nodes.make_id(name)
        section = nodes.section("", ids=[anchor], names=[anchor])
        section += nodes.title("", d["title"])
        files = ", ".join(str(mf) for mf in d["mf"])
        section += _paragraph(nodes.literal("", name), nodes.Text(f" · MF {files}"))
        section += _paragraph(nodes.Text(d["description"]))
        fields = nodes.field_list()
        for level, label in _LEVELS:
            if level in d["levels"]:
                fields += nodes.field("", nodes.field_name("", label),
                                      nodes.field_body("", _paragraph(nodes.Text(d["levels"][level]))))
        section += fields
        return section


class CovarianceChecks(_Checks):
    kind = "covariance"
    tables = staticmethod(_covariance_tables)


class DataChecks(_Checks):
    kind = "data"
    tables = staticmethod(_data_tables)


def _every_group_shown(app, env):
    shown = getattr(env, "kika_check_groups", {})
    for kind, tables in (("covariance", _covariance_tables), ("data", _data_tables)):
        if kind not in shown:
            continue
        missing = [g for g in tables()[1] if g not in shown[kind]]
        if missing:
            _LOG.warning("%s check groups on no page: %s", kind, ", ".join(missing))


def setup(app):
    app.add_directive("kika-covariance-checks", CovarianceChecks)
    app.add_directive("kika-data-checks", DataChecks)
    app.connect("env-check-consistency", _every_group_shown)
    return {"parallel_read_safe": False, "parallel_write_safe": True}
