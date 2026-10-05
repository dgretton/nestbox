from .torch_aligner import AdamAligner, GradientAligner
from .gauss_newton_aligner import GaussNewtonAligner

class AlignerFactory:
    @staticmethod
    def create_aligner(aligner_config):
        aligner_type = aligner_config['type']
        if aligner_type == 'gauss_newton':
            return GaussNewtonAligner(convergence_time=aligner_config.get('convergence_time', 1.0),
                                      step_interval=aligner_config.get('step_interval', 0.02),
                                      damping=aligner_config.get('damping', 1e-9))
        lr = aligner_config['learning_rate']
        datatype = aligner_config.get('datatype', 'float64')
        device = aligner_config.get('device', 'cpu')
        if aligner_type == 'adam':
            aligner = AdamAligner(learning_rate=lr,
                                beta1=aligner_config['beta1'],
                                beta2=aligner_config['beta2'],
                                temperature=aligner_config['temperature'],
                                datatype=datatype,
                                device=device)
        elif aligner_type == 'gradient':
            return GradientAligner(learning_rate=lr, datatype=datatype, device=device)
        else:
            raise ValueError(f"Invalid aligner type: {aligner_type}")
        return aligner
