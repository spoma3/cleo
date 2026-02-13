from brian2 import uM, ms

CalciumBindingActivationDict = {
    ('gcamp6f') : {'kon' : 66, 'koff' : 10.393, 'Kd' : 0.088 * uM, 'ton' : 10 * ms, 'toff' : 63 * ms, 'n' : 3.4, 'dFFmax' : 12.16},
    ('gcamp6s') : {'kon' : 324, 'koff' : 1.971296, 'Kd' : 0.054 * uM, 'ton' : 2 * ms, 'toff' : 346 * ms, 'n' : 3.0, 'dFFmax' : 20},
    ('gcamp3') : {'dFFmax' : 5.86},
}

# https://static-content.springer.com/esm/art%3A10.1038%2Fsrep38276/MediaObjects/41598_2016_BFsrep38276_MOESM1_ESM.pdf