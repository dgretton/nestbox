from .torch_aligner import AdamAligner, GradientAligner

class AlignerFactory:
    @staticmethod
    def create_aligner(aligner_config):
        aligner_type = aligner_config['type']
        if aligner_type == 'adam':
            aligner = AdamAligner(beta1=aligner_config['beta1'], beta2=aligner_config['beta2'])
        elif aligner_type == 'gradient':
            return GradientAligner()
        else:
            raise ValueError(f"Invalid aligner type: {aligner_type}")
        return aligner
