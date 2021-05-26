# -*- coding: utf-8 -*-

import cantera as ct
cti_str = """
units(length='cm', time='s', quantity='mol', act_energy='cal/mol')
ideal_gas(
    name='gas',
    elements='C',
    species='A B',
    reactions='all',
    initial_state=state(temperature=300.0, pressure=OneAtm)
)

species(
    name='A',
    atoms='C:12',
    thermo=(
        NASA(
            [300.00, 1000.00],
            [2.08692170E+00,  1.33149650E-01, -8.11574520E-05,
             2.94092860E-08, -6.51952130E-12, -3.59128140E+04,
             2.73552890E+01]
        ),
        NASA(
            [1000.00, 5000.00],
            [2.48802010E+01,  7.82500480E-02, -3.15509730E-05,
             5.78789000E-09, -3.98279680E-13, -4.31106840E+04,
             -9.36552550E+01]
        )
    ),
)

species(
    name='B',
    atoms='C:12',
    thermo=(
        NASA(
            [300.00, 1000.00],
            [2.08692170E+00,  1.33149650E-01, -8.11574520E-05,
             2.94092860E-08, -6.51952130E-12, -3.59128140E+04,
             2.73552890E+01]
        ),
        NASA(
            [1000.00, 5000.00],
            [2.48802010E+01,  7.82500480E-02, -3.15509730E-05,
             5.78789000E-09, -3.98279680E-13, -4.31106840E+04,
             -9.36552550E+01]
        )
    ),
)

reaction(
    equation='A => B',
    kf=Arrhenius(A=52499925000.0, b=1.5, E=40900.0),
)
"""
gas = ct.Solution(source=cti_str)
gas.TP = 1000, 101325
print(gas.reactions()[0].rate(1000))
print(gas.forward_rate_constants)

r = ct.ElementaryReaction({'A':1}, {'B':1})
r.rate = ct.Arrhenius(3.87e1, 2.7, 6260*1000*4.184)

gas.modify_reaction(0, r)
print(gas.reactions()[0].rate(1000))
print(gas.forward_rate_constants)


