"""FUDGE's ``applicationData/institution[@label='LLNL']/ENDFconversionFlags``, read and written.

The node is FUDGE's, not kika's: kika reads it so a GNDS file FUDGE wrote can be
turned back into ENDF with what FUDGE knew, and writes it back so a GNDS file
that came in with it goes out with it. An LLNL institution holding anything
else is left to the caller's loss report, untouched.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from kika.nuclear_data.model.endf_conversion import (ENDF_CONVERSION_INSTITUTION,
                                                     EndfConversionFlags)

__all__ = ["readConversionFlags", "writeConversionFlags"]

_TAG = "ENDFconversionFlags"


def readConversionFlags(applicationData: ET.Element, suite, report):
    """Pull the flags out of *applicationData* into the suite's model.

    Returns the ``applicationData`` the other readers should see: the same
    element when there were no flags, otherwise a copy without the institution
    that held them.
    """
    keep = []
    found = None
    for institution in applicationData:
        children = list(institution)
        if (institution.get("label") == ENDF_CONVERSION_INSTITUTION and children
                and all(child.tag == _TAG for child in children)):
            for child in children:
                found = found or EndfConversionFlags()
                for conversion in child:
                    if conversion.tag != "conversion":
                        report.lost(f"{_TAG}/{conversion.tag} is not modelled")
                        continue
                    found.conversions.append((conversion.get("href", ""),
                                              conversion.get("flags", "")))
            continue
        keep.append(institution)
    if found is None:
        return applicationData
    suite.applicationData.entries.append(found)
    rest = ET.Element(applicationData.tag, applicationData.attrib)
    rest.extend(keep)
    return rest


def writeConversionFlags(root: ET.Element, suite) -> None:
    """Append the suite's flags to *root*'s ``applicationData``, creating it if needed."""
    flags = EndfConversionFlags.of(suite)
    if not flags:
        return
    application = root.find("applicationData")
    if application is None:
        application = ET.SubElement(root, "applicationData")
    institution = ET.SubElement(application, "institution",
                                label=ENDF_CONVERSION_INSTITUTION)
    node = ET.SubElement(institution, _TAG)
    for href, text in flags.conversions:
        ET.SubElement(node, "conversion", flags=text, href=href)
