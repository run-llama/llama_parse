import json

import click
from llama_parse.base import LlamaParse, ResultType


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    'file',
    type=click.Path(exists=True),
)
@click.option(
    '--api-key',
    help='Defaults to $LLAMA_CLOUD_API_KEY',
    envvar='LLAMA_CLOUD_API_KEY',
    metavar='<api-key>',
)
@click.option(
    '--vendor-multimodal-model-name',
    metavar='<model>',
)
@click.option(
    '--vendor-multimodal-api-key',
    metavar='<vendor-api-key>',
)
@click.option('--invalidate-cache', is_flag=True, default=False)
@click.option(
    '--result-type',
    type=click.Choice([e for e in ResultType]),
    default=ResultType.MD,
    metavar='<result-type>',
)
def parse(**kwargs):
    """
    Parse the given file and output the result to the STDOUT

    All supported arguments match those of the LlamaParse constructor. Please
    refer to the official documentation for more information.
    """
    parser = LlamaParse(
        result_type=kwargs['result_type'],
        api_key=kwargs['api_key'],
        use_vendor_multimodal_model=(
            kwargs['vendor_multimodal_model_name'] is not None
        ),
        vendor_multimodal_model_name=kwargs['vendor_multimodal_model_name'],
        vendor_multimodal_api_key=kwargs['vendor_multimodal_api_key'],
        invalidate_cache=kwargs['invalidate_cache'],
        verbose=False,
        split_by_page=False,
        ignore_errors=False,
    )
    click.echo(
        json.dumps(parser.get_json_result(kwargs['file']))
        if kwargs['result_type'] == ResultType.JSON
        else parser.load_data(kwargs['file'])[0].text
    )


if __name__ == '__main__':
    cli()
