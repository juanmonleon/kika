"""Heat a synthetic scalar curve; no external input files required."""
from kika.processing.doppler import (
    DopplerContext, DopplerOptions, SupportPolicy,
    prepare_curve, evaluate_broadened, broaden_curve,
)

source = prepare_curve(
    (([0.001, 2.0], [2.0, 2.0]),),
    context=DopplerContext(56.0, mass_origin="synthetic-example"),
    source_temperature=0.0,
    support=SupportPolicy(lower="constant", upper="constant"),
)
options = DopplerOptions(rtol=1e-3, atol=1e-8)
pointwise = evaluate_broadened(source, [0.01, 0.1, 1.0],
                              temperature=600.0, options=options)
heated = broaden_curve(source, temperature=600.0, options=options)
print(pointwise.values)
print(heated.form)
print(heated.verify_table())
