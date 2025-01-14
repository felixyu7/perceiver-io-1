from pytorch_lightning.utilities.cli import LightningArgumentParser

from perceiver import LitTextClassifier
from scripts import CLI

# register data module via import
from data import IMDBDataModule


class TextClassifierCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments('data.vocab_size', 'model.vocab_size', apply_on='instantiate')
        parser.link_arguments('data.max_seq_len', 'model.max_seq_len', apply_on='instantiate')
        parser.set_defaults({
            'experiment': 'seq_clf',
            'model.num_classes': 2,
            'model.num_latents': 64,
            'model.num_latent_channels': 64,
            'model.num_encoder_layers': 3,
            'model.num_decoder_cross_attention_heads': 1,
        })


if __name__ == '__main__':
    TextClassifierCLI(LitTextClassifier, description='Text classifier', run=True)
