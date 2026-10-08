"""Versioned, opt-in KIKA resonance data in GNDS applicationData.

This is application-specific data, not an extension to the standard channel
schema. A generic GNDS consumer must not reconstruct these parameters after
ignoring the institution. The ordinary writer continues to report the gap.

The institution carries ``resonanceChannelFunctions`` (complex KPS,
tabulated LBK) and ``unresolvedCrossSectionInterpolation`` — ENDF's URR INT,
which interpolates the *cross sections* between the parameter energies.
§19.4.1 has a law only for each parameter function, and FUDGE copies INT
there, which is a different calculation (resonance roadmap D13/N3).
``unresolvedPotentialScattering`` declares an independent potential policy;
``unresolvedNeutronWidthConvention`` explicitly identifies physical neutron
means instead of reduced GN0. Both are opt-in canonical KIKA conventions.
"""
from copy import deepcopy
import hashlib
import json
import xml.etree.ElementTree as ET
from kika.nuclear_data.model.resonances import RMatrix,ComplexChannelFunction
from kika.nuclear_data.model.enums import Interpolation

LABEL='KIKA::resonance_channel_functions'
URR='unresolved'
URRPOT='unresolvedPotential'
URRWIDTH='unresolvedWidth'


def _fingerprint(formalism):
    number=lambda v: None if v is None else float(v)
    data=[(g.label,number(g.spin),g.parity,list(map(float,g.energies)),[list(map(float,row)) for row in g.widths],
           [(c.label,c.resonanceReaction,c.L,number(c.channelSpin),c.columnIndex) for c in g.channels])
          for g in formalism.spinGroups]
    return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()


def _urrFingerprint(widths):
    """Identity of the URR spin groups an INT entry belongs to: L, J and D."""
    def spacing(group):
        form=group.levelSpacingFunction
        if form is not None:
            if hasattr(form,'constant'):return [float(form.constant)]
            return list(map(float,form.toEndfRegions()[1]))
        return list(map(float,group.levelSpacing)) if group.levelSpacing is not None else []
    data=[(g.L,float(g.J),spacing(g)) for g in widths.spinGroups]
    return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()


def extract(suite):
    """Copy the suite, moving only fields absent from standard GNDS channels."""
    saved=[];result=deepcopy(suite)
    if result.resonances is None:return result,saved
    for r,region in enumerate(result.resonances.resolved):
        f=region.formalism
        if not isinstance(f,RMatrix):continue
        fingerprint=_fingerprint(f)
        for g,group in enumerate(f.spinGroups):
            for c,channel in enumerate(group.channels):
                if (channel.additionalPhaseShift is None and channel.phaseAbsorptionReaction is None
                        and channel.phaseShiftMode is None and channel.tabulatedBackground is None):continue
                saved.append((r,g,c,fingerprint,deepcopy(channel)))
                channel.additionalPhaseShift=None;channel.phaseAbsorptionReaction=None
                channel.phaseShiftMode=None;channel.tabulatedBackground=None
    widths=getattr(result.resonances.unresolved,'tabulatedWidths',None)
    if widths is not None:
        fingerprint=_urrFingerprint(widths)
        if widths.potentialScatteringInterpolation is not None or widths.potentialScatteringEnergies is not None:
            saved.append((URRPOT,fingerprint,widths.potentialScatteringInterpolation,
                          deepcopy(widths.potentialScatteringEnergies)))
            widths.potentialScatteringInterpolation=None; widths.potentialScatteringEnergies=None
        for g,group in enumerate(widths.spinGroups):
            for c,channel in enumerate(group.channels):
                if channel.neutronWidthConvention!='reduced':
                    if channel.neutronWidthConvention!='physical':raise ValueError('unknown URR neutron-width convention')
                    saved.append((URRWIDTH,g,c,channel.label,fingerprint))
                    channel.neutronWidthConvention='reduced'
            if group.crossSectionInterpolation is None:continue
            grid=(group.crossSectionEnergies if group.crossSectionEnergies is not None else
                  group.levelSpacingEnergies if group.levelSpacingEnergies is not None else widths.energyGrid)
            saved.append((URR,g,group.L,float(group.J),fingerprint,group.crossSectionInterpolation,deepcopy(grid)))
            group.crossSectionInterpolation=None;group.crossSectionEnergies=None
    return result,saved


def write(root,saved,report):
    if not saved:return
    from .encode import _function
    application=ET.SubElement(root,'applicationData')
    institution=ET.SubElement(application,'institution',label=LABEL)
    channels=[s for s in saved if s[0] not in (URR,URRPOT,URRWIDTH)];urr=[s for s in saved if s[0]==URR]
    widths=[s for s in saved if s[0]==URRWIDTH]
    if widths:
        block=ET.SubElement(institution,'unresolvedNeutronWidthConvention',version='1')
        for _,g,c,label,fingerprint in widths:
            ET.SubElement(block,'width',group=str(g),channel=str(c),label=label,
                          convention='physical',parameterFingerprint=fingerprint)
    for _,fingerprint,law,grid in (s for s in saved if s[0]==URRPOT):
        if law is None:raise ValueError('URR potential grid requires a policy')
        mode='continuous' if law=='continuous' else 'interpolate'
        block=ET.SubElement(institution,'unresolvedPotentialScattering',version='1',mode=mode,
                            parameterFingerprint=fingerprint)
        if mode=='interpolate':block.set('interpolation',law.value)
        if grid is not None:ET.SubElement(block,'energies',unit='eV').text=' '.join(repr(float(v)) for v in grid)
    if urr:
        block=ET.SubElement(institution,'unresolvedCrossSectionInterpolation',
                            version='2' if any(s[-1] is not None for s in urr) else '1')
        for _,g,L,J,fingerprint,law,grid in urr:
            entry=ET.SubElement(block,'J',index=str(g),L=str(L),value=repr(J),interpolation=law.value,
                                parameterFingerprint=fingerprint)
            if grid is not None:ET.SubElement(entry,'energies',unit='eV').text=' '.join(repr(float(v)) for v in grid)
    if not channels:
        report.warn('KIKA-specific resonance applicationData written; consumers must interpret this institution before reconstructing resonances')
        return
    data=ET.SubElement(institution,'resonanceChannelFunctions',version='1')
    for r,g,c,fingerprint,ch in channels:
        attributes=dict(region=str(r),group=str(g),channel=str(c),parameterFingerprint=fingerprint)
        if ch.phaseShiftMode is not None:attributes['phaseShiftMode']=str(ch.phaseShiftMode)
        if ch.phaseAbsorptionReaction is not None:attributes['phaseAbsorptionReaction']=ch.phaseAbsorptionReaction
        entry=ET.SubElement(data,'channel',attributes)
        for name in ('additionalPhaseShift','tabulatedBackground'):
            value=getattr(ch,name)
            if value is None:continue
            holder=ET.SubElement(entry,name)
            for part in ('real','imaginary'):
                _function(ET.SubElement(holder,part),getattr(value,part),report,'KIKA resonance '+name)
    report.warn('KIKA-specific resonance applicationData written; consumers must interpret this institution before reconstructing resonances')


def read(application,suite,report,read_function):
    """Restore known typed data; leave other institutions for loss reporting."""
    unknown=[];seen=set()
    for institution in application:
        if institution.get('label')!=LABEL:
            unknown.append(institution.get('label',institution.tag));continue
        tags=[child.tag for child in institution]
        if not tags or len(set(tags))!=len(tags) or set(tags)-{'resonanceChannelFunctions','unresolvedCrossSectionInterpolation','unresolvedPotentialScattering','unresolvedNeutronWidthConvention'}:
            raise ValueError('unknown KIKA resonance applicationData content')
        if any(child.get('version') not in ({'1','2'} if child.tag=='unresolvedCrossSectionInterpolation' else {'1'}) for child in institution):
            report.unsupportedNode('unsupported KIKA resonance applicationData version');continue
        if any(set(child.attrib)!={'version'} for child in institution if child.tag!='unresolvedPotentialScattering'):
            raise ValueError('unknown KIKA resonance applicationData content')
        block=institution.find('unresolvedCrossSectionInterpolation')
        if block is not None:_readUnresolved(block,suite)
        block=institution.find('unresolvedPotentialScattering')
        if block is not None:_readUnresolvedPotential(block,suite)
        block=institution.find('unresolvedNeutronWidthConvention')
        if block is not None:_readUnresolvedWidths(block,suite)
        data=institution.find('resonanceChannelFunctions')
        for entry in (data if data is not None else []):
            if entry.tag!='channel':raise ValueError('unknown KIKA resonance applicationData entry')
            if set(entry.attrib)-{'region','group','channel','parameterFingerprint','phaseShiftMode','phaseAbsorptionReaction'}:
                raise ValueError('unknown KIKA resonance channel attribute')
            if any(child.tag not in ('additionalPhaseShift','tabulatedBackground') for child in entry):
                raise ValueError('unknown KIKA resonance channel function')
            if len({child.tag for child in entry})!=len(entry):
                raise ValueError('duplicate KIKA resonance channel function')
            key=tuple(int(entry.attrib[k]) for k in ('region','group','channel'))
            if min(key)<0 or key in seen:raise ValueError('invalid or duplicate KIKA resonance channel reference')
            seen.add(key)
            try:
                region=suite.resonances.resolved[key[0]];f=region.formalism
                channel=f.spinGroups[key[1]].channels[key[2]]
            except (AttributeError,IndexError) as exc:raise ValueError('unresolved KIKA resonance channel reference') from exc
            if entry.get('parameterFingerprint')!=_fingerprint(f):
                raise ValueError('KIKA resonance applicationData does not match its parameter table')
            mode=entry.get('phaseShiftMode')
            channel.phaseShiftMode=None if mode is None else int(mode)
            channel.phaseAbsorptionReaction=entry.get('phaseAbsorptionReaction')
            for name in ('additionalPhaseShift','tabulatedBackground'):
                holder=entry.find(name)
                if holder is None:continue
                if holder.attrib or sorted(child.tag for child in holder)!=['imaginary','real']:
                    raise ValueError('invalid KIKA complex channel function components')
                components=[]
                for part in ('real','imaginary'):
                    child=holder.find(part)
                    if child is None or len(child)!=1:raise ValueError('incomplete KIKA complex channel function')
                    value=read_function(child[0],'/reactionSuite/applicationData/'+name,part)
                    if value is None:raise ValueError('unsupported KIKA complex channel function')
                    components.append(value)
                setattr(channel,name,ComplexChannelFunction(*components))
        report.warn('KIKA-specific resonance applicationData interpreted; these resonance fields are not standard GNDS nodes')
    return unknown


def _readUnresolved(block,suite):
    widths=getattr(getattr(suite.resonances,'unresolved',None),'tabulatedWidths',None) if suite.resonances is not None else None
    if widths is None:raise ValueError('KIKA URR interpolation without an unresolved region')
    fingerprint=_urrFingerprint(widths);seen=set()
    for entry in block:
        if entry.tag!='J' or set(entry.attrib)!={'index','L','value','interpolation','parameterFingerprint'}:
            raise ValueError('unknown KIKA URR interpolation entry')
        g=int(entry.get('index'))
        if g<0 or g>=len(widths.spinGroups) or g in seen:
            raise ValueError('invalid or duplicate KIKA URR interpolation reference')
        seen.add(g);group=widths.spinGroups[g]
        if entry.get('parameterFingerprint')!=fingerprint or int(entry.get('L'))!=group.L or float(entry.get('value'))!=float(group.J):
            raise ValueError('KIKA URR interpolation does not match its spin groups')
        group.crossSectionInterpolation=Interpolation(entry.get('interpolation'))
        if len(entry)>1 or any(child.tag!='energies' or child.attrib!={'unit':'eV'} or len(child) for child in entry):
            raise ValueError('unknown KIKA URR sigma grid')
        if len(entry):
            import numpy as np
            grid=np.asarray([float(v) for v in (entry[0].text or '').split()])
            # Preserve ENDF's repeated nodes, including LSSF=1 parameter
            # discontinuities. Processing validates the selected sigma policy.
            if len(grid)<2 or not np.all(np.isfinite(grid)) or np.any(grid<=0) or np.any(np.diff(grid)<0):
                raise ValueError('invalid KIKA URR sigma grid')
            group.crossSectionEnergies=grid


def _readUnresolvedPotential(block,suite):
    import numpy as np
    widths=getattr(getattr(suite.resonances,'unresolved',None),'tabulatedWidths',None) if suite.resonances is not None else None
    if widths is None:raise ValueError('KIKA URR potential policy without an unresolved region')
    mode=block.get('mode')
    required={'version','mode','parameterFingerprint'}|({'interpolation'} if mode=='interpolate' else set())
    if mode not in ('continuous','interpolate') or set(block.attrib)!=required:
        raise ValueError('unknown KIKA URR potential policy')
    if block.get('parameterFingerprint')!=_urrFingerprint(widths):
        raise ValueError('KIKA URR potential policy does not match its spin groups')
    if len(block)>1 or any(child.tag!='energies' or child.attrib!={'unit':'eV'} or len(child) for child in block):
        raise ValueError('unknown KIKA URR potential grid')
    grid=None if not len(block) else np.asarray([float(v) for v in (block[0].text or '').split()])
    if grid is not None and (grid.ndim!=1 or len(grid)<2 or not np.all(np.isfinite(grid)) or np.any(grid<=0) or np.any(np.diff(grid)<=0)):
        raise ValueError('invalid KIKA URR potential grid')
    if mode=='continuous' and grid is not None:raise ValueError('continuous URR potential does not use a grid')
    widths.potentialScatteringInterpolation='continuous' if mode=='continuous' else Interpolation(block.get('interpolation'))
    widths.potentialScatteringEnergies=grid


def _readUnresolvedWidths(block,suite):
    widths=getattr(getattr(suite.resonances,'unresolved',None),'tabulatedWidths',None) if suite.resonances is not None else None
    if widths is None:raise ValueError('KIKA URR width convention without an unresolved region')
    seen=set();fingerprint=_urrFingerprint(widths)
    for entry in block:
        if entry.tag!='width' or len(entry) or set(entry.attrib)!={'group','channel','label','convention','parameterFingerprint'}:
            raise ValueError('unknown KIKA URR neutron-width convention')
        g,c=int(entry.get('group')),int(entry.get('channel'))
        if min(g,c)<0 or (g,c) in seen:raise ValueError('invalid or duplicate KIKA URR width reference')
        seen.add((g,c))
        try:channel=widths.spinGroups[g].channels[c]
        except IndexError as exc:raise ValueError('invalid KIKA URR width reference') from exc
        if entry.get('parameterFingerprint')!=fingerprint or entry.get('label')!=channel.label:
            raise ValueError('KIKA URR width convention does not match its channels')
        if entry.get('convention')!='physical':raise ValueError('unknown KIKA URR neutron-width convention')
        channel.neutronWidthConvention='physical'
