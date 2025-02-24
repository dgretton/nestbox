from .torch_aligner import AdamAligner, GradientAligner

class AlignerFactory:
    @staticmethod
    def create_aligner(aligner_config):
        aligner_type = aligner_config['type']
        lr = aligner_config['learning_rate']
        datatype = aligner_config.get('datatype', 'float64')
        if aligner_type == 'adam':
            aligner = AdamAligner(learning_rate=lr,
                                beta1=aligner_config['beta1'],
                                beta2=aligner_config['beta2'],
                                temperature=aligner_config['temperature'],
                                datatype=datatype)
        elif aligner_type == 'gradient':
            return GradientAligner(learning_rate=lr, datatype=datatype)
        else:
            raise ValueError(f"Invalid aligner type: {aligner_type}")
        return aligner
