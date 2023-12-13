from amisc.rv import UniformRV, NormalRV, LogNormalRV, LogUniformRV, ScalarRV


def test_loading_rvs():
    """Test random variable construction and methods."""
    rvs = [UniformRV(0, 1, id='x'), NormalRV(0, 1, id='y'), LogUniformRV(1, 2, id='z'), LogNormalRV(2, 1, id='a'),
           ScalarRV(id='h')]
    samples_pdf = []
    samples_domain = []
    pdf_vals = []
    labels = []
    for v in rvs:
        sample = v.sample(5)
        samples_pdf.append(sample)
        samples_domain.append(v.sample_domain(5))
        pdf_vals.append(v.pdf(sample))
        labels.append(v.to_tex(symbol=False))
        v.update_bounds(*v.bds)


def test_rv_sampling():
    # TODO: make sure rvs are sampling and evaluating correctly
    pass
