"""Read and write modeled documentation without discarding free text."""
import xml.etree.ElementTree as ET
from kika.nuclear_data.model.documentation import Documentation,DocumentationText,Author,Date


def readDocumentation(element, report):
    def attributes(node, allowed):
        for name in node.attrib:
            if name not in allowed:
                report.lost(f'documentation/{node.tag}: attribute {name!r} is not modeled')
        return {name:value for name,value in node.attrib.items() if name in allowed}
    if (element.text or "").strip():
        report.lost("documentation: text outside modeled children is not supported")
    document=Documentation(**attributes(element,('doi','publicationDate','version')))
    seen=set()
    for child in element:
        if child.tag in seen:
            raise ValueError(f'duplicate documentation child {child.tag}')
        seen.add(child.tag)
        if child.tag in ('title','abstract','body','endfCompatible'):
            if len(child):
                report.lost(f'documentation/{child.tag}: nested text markup is not modeled')
            setattr(document,child.tag,DocumentationText(''.join(child.itertext()),
                **attributes(child,('encoding','markup','label'))))
        elif child.tag in ('authors','dates'):
            attributes(child,())
            for entry in child:
                if child.tag=='authors' and entry.tag=='author':
                    document.authors.append(Author(**attributes(entry,('name','orcid','email'))))
                elif child.tag=='dates' and entry.tag=='date':
                    document.dates.append(Date(**attributes(entry,('value','dateType'))))
                else:
                    report.lost(f'documentation/{child.tag}: {entry.tag} is not modeled')
                if len(entry):
                    report.lost(f'documentation/{child.tag}/{entry.tag}: child metadata is not modeled')
        else:
            report.lost(f'documentation/{child.tag}: collection is not modeled')
    return document


def writeDocumentation(parent, document):
    element=ET.SubElement(parent,'documentation')
    def attributes(node, source, names):
        for name in names:
            value=getattr(source,name)
            if value is not None:node.set(name,value)
    attributes(element,document,('doi','publicationDate','version'))
    for name in ('authors','dates'):
        entries=getattr(document,name)
        if entries:
            wrapper=ET.SubElement(element,name)
            for entry in entries:
                node=ET.SubElement(wrapper,'author' if name=='authors' else 'date')
                attributes(node,entry,('name','orcid','email') if name=='authors' else ('value','dateType'))
    for name in ('title','abstract','body','endfCompatible'):
        text=getattr(document,name)
        if text is not None:
            node=ET.SubElement(element,name)
            attributes(node,text,('encoding','markup','label'))
            node.text=text.text
    return element
