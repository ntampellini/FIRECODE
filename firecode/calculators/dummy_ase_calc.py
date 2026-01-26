import ase.calculators.calculator
import numpy as np

class DummyCalculator(ase.calculators.calculator.Calculator):
    """
    Dummy ASE calculator returning zeros.

    """

    implemented_properties = ['energy', 'forces']

    def __init__(self):
        # pass
        super().__init__()
        # self.model = model
        # self.charge = charge
        # self.device = next(model.parameters()).device
        # cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
        # self.cutoff = float(cutoff)
        # self._t_numbers = None
        # self._t_charge = None

    # def do_reset(self):
    #     self._t_numbers = None
    #     self._t_charge = None
    #     self.charge = 0.0

    # def set_charge(self, charge):
    #     self.charge = float(charge)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        
        super().calculate(atoms, properties, system_changes)
        do_forces = 'forces' in properties

        self.results['energy'] = 0.0

        if do_forces:
            n_atoms = len(atoms.positions)
            self.results['forces'] = np.zeros((n_atoms, 3))

