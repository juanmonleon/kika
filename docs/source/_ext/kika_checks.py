"""``kika-covariance-checks``: one group of ``CHECKS``, the texts the reports and the app show.

Nodes are built directly: the texts hold ``|rho|``, a table in Markdown and a substitution in rST.
"""
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

_LEVELS = (("defect", "Defect"), ("warn", "Warning"), ("note", "Note"))
_LOG = logging.getLogger(__name__)


def _paragraph(*children):
    return nodes.paragraph("", "", *children)


class CovarianceChecks(Directive):
    required_arguments = 1

    def run(self):
        from kika.endf.checks.descriptions import CHECKS, GROUPS

        group = self.arguments[0]
        if group not in GROUPS:
            raise self.error(f"no covariance check group {group!r}; the groups are {', '.join(GROUPS)}")
        env = self.state.document.settings.env
        env.kika_check_groups = getattr(env, "kika_check_groups", set()) | {group}
        return [self._section(name, CHECKS[name].to_dict()) for name in GROUPS[group]]

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


def _every_group_shown(app, env):
    from kika.endf.checks.descriptions import GROUPS

    missing = [g for g in GROUPS if g not in getattr(env, "kika_check_groups", set())]
    if missing and getattr(env, "kika_check_groups", None):
        _LOG.warning("covariance check groups on no page: %s", ", ".join(missing))


def setup(app):
    app.add_directive("kika-covariance-checks", CovarianceChecks)
    app.connect("env-check-consistency", _every_group_shown)
    return {"parallel_read_safe": False, "parallel_write_safe": True}
