import os
import asyncio
from urllib.parse import urlparse

import httpx
import mimetypes
import time
from pathlib import Path, PurePath, PurePosixPath
from typing import AsyncGenerator, Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from io import BufferedIOBase

from fsspec import AbstractFileSystem
from llama_index.core.async_utils import asyncio_run, run_jobs
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.constants import DEFAULT_BASE_URL
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.readers.file.base import get_default_fs
from llama_index.core.schema import Document
from llama_parse.utils import (
    nest_asyncio_err,
    nest_asyncio_msg,
    ResultType,
    SUPPORTED_FILE_TYPES,
)
from copy import deepcopy

# can put in a path to the file or the file bytes itself
# if passing as bytes or a buffer, must provide the file_name in extra_info
FileInput = Union[str, bytes, BufferedIOBase]

_DEFAULT_SEPARATOR = "\n---\n"


class LlamaParse(BasePydanticReader):
    """A smart-parser for files."""

    # Library / access specific configurations
    api_key: str = Field(
        default="",
        description="The API key for the LlamaParse API.",
        validate_default=True,
    )
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        description="The base URL of the Llama Parsing API.",
    )
    check_interval: int = Field(
        default=1,
        description="The interval in seconds to check if the parsing is done.",
    )
    custom_client: Optional[httpx.AsyncClient] = Field(
        default=None, description="A custom HTTPX client to use for sending requests."
    )
    ignore_errors: bool = Field(
        default=True,
        description="Whether or not to ignore and skip errors raised during parsing.",
    )
    max_timeout: int = Field(
        default=2000,
        description="The maximum timeout in seconds to wait for the parsing to finish.",
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=10,
        description="The number of workers to use sending API requests for parsing.",
    )
    result_type: ResultType = Field(
        default=ResultType.TXT, description="The result type for the parser."
    )
    show_progress: bool = Field(
        default=True, description="Show progress when parsing multiple files."
    )
    split_by_page: bool = Field(
        default=True,
        description="Whether to split by page using the page separator",
    )
    verbose: bool = Field(
        default=True, description="Whether to print the progress of the parsing."
    )

    # Parsing specific configurations (Alphabetical order)
    annotate_links: Optional[bool] = Field(
        default=False,
        description="Annotate links found in the document to extract their URL.",
    )
    auto_mode: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will automatically select the best mode to extract text from documents based on the rules provide. Will use the 'accurate' default mode by default and will upgrade page that match the rule to Premium mode.",
    )
    auto_mode_trigger_on_image_in_page: Optional[bool] = Field(
        default=False,
        description="If auto_mode is set to true, the parser will upgrade the page that contain an image to Premium mode.",
    )
    auto_mode_trigger_on_table_in_page: Optional[bool] = Field(
        default=False,
        description="If auto_mode is set to true, the parser will upgrade the page that contain a table to Premium mode.",
    )
    auto_mode_trigger_on_text_in_page: Optional[str] = Field(
        default=None,
        description="If auto_mode is set to true, the parser will upgrade the page that contain the text to Premium mode.",
    )
    auto_mode_trigger_on_regexp_in_page: Optional[str] = Field(
        default=None,
        description="If auto_mode is set to true, the parser will upgrade the page that match the regexp to Premium mode.",
    )
    azure_openai_api_version: Optional[str] = Field(
        default=None, description="Azure Openai API Version"
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None, description="Azure Openai Deployment Name"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure Openai Endpoint"
    )
    azure_openai_key: Optional[str] = Field(
        default=None, description="Azure Openai Key"
    )
    bbox_bottom: Optional[float] = Field(
        default=None,
        description="The bottom margin of the bounding box to use to extract text from documents expressed as a float between 0 and 1 representing the percentage of the page height.",
    )
    bbox_left: Optional[float] = Field(
        default=None,
        description="The left margin of the bounding box to use to extract text from documents expressed as a float between 0 and 1 representing the percentage of the page width.",
    )
    bbox_right: Optional[float] = Field(
        default=None,
        description="The right margin of the bounding box to use to extract text from documents expressed as a float between 0 and 1 representing the percentage of the page width.",
    )
    bbox_top: Optional[float] = Field(
        default=None,
        description="The top margin of the bounding box to use to extract text from documents expressed as a float between 0 and 1 representing the percentage of the page height.",
    )
    continuous_mode: Optional[bool] = Field(
        default=False,
        description="Parse documents continuously, leading to better results on documents where tables span across two pages.",
    )
    disable_ocr: Optional[bool] = Field(
        default=False,
        description="Disable the OCR on the document. LlamaParse will only extract the copyable text from the document.",
    )
    disable_image_extraction: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will not extract images from the document. Make the parser faster.",
    )
    do_not_cache: Optional[bool] = Field(
        default=False,
        description="If set to true, the document will not be cached. This mean that you will be re-charged it you reprocess them as they will not be cached.",
    )
    do_not_unroll_columns: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will keep column in the text according to document layout. Reduce reconstruction accuracy, and LLM's/embedings performances in most case.",
    )
    extract_charts: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will extract/tag charts from the document.",
    )
    fast_mode: Optional[bool] = Field(
        default=False,
        description="Note: Non compatible with gpt-4o. If set to true, the parser will use a faster mode to extract text from documents. This mode will skip OCR of images, and table/heading reconstruction.",
    )
    guess_xlsx_sheet_names: Optional[bool] = Field(
        default=False,
        description="Whether to guess the sheet names of the xlsx file.",
    )
    html_make_all_elements_visible: Optional[bool] = Field(
        default=False,
        description="If set to true, when parsing HTML the parser will consider all elements display not element as display block.",
    )
    html_remove_fixed_elements: Optional[bool] = Field(
        default=False,
        description="If set to true, when parsing HTML the parser will remove fixed elements. Useful to hide cookie banners.",
    )
    html_remove_navigation_elements: Optional[bool] = Field(
        default=False,
        description="If set to true, when parsing HTML the parser will remove navigation elements. Useful to hide menus, header, footer.",
    )
    http_proxy: Optional[str] = Field(
        default=None,
        description="(optional) If set with input_url will use the specified http proxy to download the file.",
    )
    invalidate_cache: Optional[bool] = Field(
        default=False,
        description="If set to true, the cache will be ignored and the document re-processes. All document are kept in cache for 48hours after the job was completed to avoid processing the same document twice.",
    )
    is_formatting_instruction: Optional[bool] = Field(
        default=False,
        description="Allow the parsing instruction to also format the output. Disable to have a cleaner markdown output.",
    )
    language: Optional[str] = Field(
        default="en", description="The language of the text to parse."
    )
    max_pages: Optional[int] = Field(
        default=None,
        description="The maximum number of pages to extract text from documents. If set to 0 or not set, all pages will be that should be extracted will be extracted (can work in combination with targetPages).",
    )
    output_pdf_of_document: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will also output a PDF of the document. (except for spreadsheets)",
    )
    output_s3_path_prefix: Optional[str] = Field(
        default=None,
        description="An S3 path prefix to store the output of the parsing job. If set, the parser will upload the output to S3. The bucket need to be accessible from the LlamaIndex organization.",
    )
    page_prefix: Optional[str] = Field(
        default=None,
        description="A templated prefix to add to the beginning of each page. If it contain `{page_number}`, it will be replaced by the page number.",
    )
    page_separator: Optional[str] = Field(
        default=None,
        description="A templated  page separator to use to split the text.  If it contain `{page_number}`,it will be replaced by the next page number. If not set will the default separator '\\n---\\n' will be used.",
    )
    page_suffix: Optional[str] = Field(
        default=None,
        description="A templated suffix to add to the beginning of each page. If it contain `{page_number}`, it will be replaced by the page number.",
    )
    parsing_instruction: Optional[str] = Field(
        default="", description="The parsing instruction for the parser."
    )
    premium_mode: Optional[bool] = Field(
        default=False,
        description="Use our best parser mode if set to True.",
    )
    skip_diagonal_text: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will ignore diagonal text (when the text rotation in degrees modulo 90 is not 0).",
    )
    structured_output: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will output structured data based on the provided JSON Schema.",
    )
    structured_output_json_schema: Optional[str] = Field(
        default=None,
        description="A JSON Schema to use to structure the output of the parsing job. If set, the parser will output structured data based on the provided JSON Schema.",
    )
    structured_output_json_schema_name: Optional[str] = Field(
        default=None,
        description="The named JSON Schema to use to structure the output of the parsing job. For convenience / testing, LlamaParse provides a few named JSON Schema that can be used directly. Use 'imFeelingLucky' to let llamaParse dream the schema.",
    )
    take_screenshot: Optional[bool] = Field(
        default=False,
        description="Whether to take screenshot of each page of the document.",
    )
    target_pages: Optional[str] = Field(
        default=None,
        description="The target pages to extract text from documents. Describe as a comma separated list of page numbers. The first page of the document is page 0",
    )
    use_vendor_multimodal_model: Optional[bool] = Field(
        default=False,
        description="Whether to use the vendor multimodal API.",
    )
    vendor_multimodal_api_key: Optional[str] = Field(
        default=None,
        description="The API key for the multimodal API.",
    )
    vendor_multimodal_model_name: Optional[str] = Field(
        default=None,
        description="The model name for the vendor multimodal API.",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="A URL that needs to be called at the end of the parsing job.",
    )

    # Deprecated
    bounding_box: Optional[str] = Field(
        default=None,
        description="The bounding box to use to extract text from documents describe as a string containing the bounding box margins",
    )
    gpt4o_mode: Optional[bool] = Field(
        default=False,
        description="Whether to use gpt-4o extract text from documents.",
    )
    gpt4o_api_key: Optional[str] = Field(
        default=None,
        description="The API key for the GPT-4o API. Lowers the cost of parsing.",
    )

    @field_validator("api_key", mode="before", check_fields=True)
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            import os

            api_key = os.getenv("LLAMA_CLOUD_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")
            return api_key

        return v

    @field_validator("base_url", mode="before", check_fields=True)
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate the base URL."""
        url = os.getenv("LLAMA_CLOUD_BASE_URL", None)
        return url or v or DEFAULT_BASE_URL

    @asynccontextmanager
    async def client_context(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Create a context for the HTTPX client."""
        if self.custom_client is not None:
            yield self.custom_client
        else:
            async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                yield client

    def _is_input_url(self, file_path: FileInput) -> bool:
        """Check if the input is a valid URL.

        This method checks for:
        - Proper URL scheme (http/https)
        - Valid URL structure
        - Network location (domain)
        """
        if not isinstance(file_path, str):
            return False
        try:
            result = urlparse(file_path)
            return all(
                [
                    result.scheme in ("http", "https"),
                    result.netloc,  # Has domain
                    result.scheme,  # Has scheme
                ]
            )
        except Exception:
            return False

    def _is_s3_url(self, file_path: FileInput) -> bool:
        """Check if the input is a valid URL.

        This method checks for:
        - Proper S3 scheme (s3://)
        """
        if isinstance(file_path, str):
            return file_path.startswith("s3://")
        return False

    # upload a document and get back a job_id
    async def _create_job(
        self,
        file_input: FileInput,
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/api/parsing/upload"
        files = None
        file_handle = None
        input_url = file_input if self._is_input_url(file_input) else None
        input_s3_path = file_input if self._is_s3_url(file_input) else None

        if isinstance(file_input, (bytes, BufferedIOBase)):
            if not extra_info or "file_name" not in extra_info:
                raise ValueError(
                    "file_name must be provided in extra_info when passing bytes"
                )
            file_name = extra_info["file_name"]
            mime_type = mimetypes.guess_type(file_name)[0]
            files = {"file": (file_name, file_input, mime_type)}
        elif input_url is not None:
            files = None
        elif input_s3_path is not None:
            files = None
        elif isinstance(file_input, (str, Path, PurePosixPath, PurePath)):
            file_path = str(file_input)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in SUPPORTED_FILE_TYPES:
                raise Exception(
                    f"Currently, only the following file types are supported: {SUPPORTED_FILE_TYPES}\n"
                    f"Current file type: {file_ext}"
                )
            mime_type = mimetypes.guess_type(file_path)[0]
            # Open the file here for the duration of the async context
            # load data, set the mime type
            fs = fs or get_default_fs()
            file_handle = fs.open(file_input, "rb")
            files = {"file": (os.path.basename(file_path), file_handle, mime_type)}
        else:
            raise ValueError(
                "file_input must be either a file path string, file bytes, or buffer object"
            )

        data: Dict[str, Any] = {}

        data["from_python_package"] = True

        if self.annotate_links:
            data["annotate_links"] = self.annotate_links

        if self.auto_mode:
            data["auto_mode"] = self.auto_mode

        if self.auto_mode_trigger_on_image_in_page:
            data[
                "auto_mode_trigger_on_image_in_page"
            ] = self.auto_mode_trigger_on_image_in_page

        if self.auto_mode_trigger_on_table_in_page:
            data[
                "auto_mode_trigger_on_table_in_page"
            ] = self.auto_mode_trigger_on_table_in_page

        if self.auto_mode_trigger_on_text_in_page is not None:
            data[
                "auto_mode_trigger_on_text_in_page"
            ] = self.auto_mode_trigger_on_text_in_page

        if self.auto_mode_trigger_on_regexp_in_page is not None:
            data[
                "auto_mode_trigger_on_regexp_in_page"
            ] = self.auto_mode_trigger_on_regexp_in_page

        if self.azure_openai_api_version is not None:
            data["azure_openai_api_version"] = self.azure_openai_api_version

        if self.azure_openai_deployment_name is not None:
            data["azure_openai_deployment_name"] = self.azure_openai_deployment_name

        if self.azure_openai_endpoint is not None:
            data["azure_openai_endpoint"] = self.azure_openai_endpoint

        if self.azure_openai_key is not None:
            data["azure_openai_key"] = self.azure_openai_key

        if self.bbox_bottom is not None:
            data["bbox_bottom"] = self.bbox_bottom

        if self.bbox_left is not None:
            data["bbox_left"] = self.bbox_left

        if self.bbox_right is not None:
            data["bbox_right"] = self.bbox_right

        if self.bbox_top is not None:
            data["bbox_top"] = self.bbox_top

        if self.continuous_mode:
            data["continuous_mode"] = self.continuous_mode

        if self.disable_ocr:
            data["disable_ocr"] = self.disable_ocr

        if self.disable_image_extraction:
            data["disable_image_extraction"] = self.disable_image_extraction

        if self.do_not_cache:
            data["do_not_cache"] = self.do_not_cache

        if self.do_not_unroll_columns:
            data["do_not_unroll_columns"] = self.do_not_unroll_columns

        if self.extract_charts:
            data["extract_charts"] = self.extract_charts

        if self.fast_mode:
            data["fast_mode"] = self.fast_mode

        if self.guess_xlsx_sheet_names:
            data["guess_xlsx_sheet_names"] = self.guess_xlsx_sheet_names

        if self.html_make_all_elements_visible:
            data["html_make_all_elements_visible"] = self.html_make_all_elements_visible

        if self.html_remove_fixed_elements:
            data["html_remove_fixed_elements"] = self.html_remove_fixed_elements

        if self.html_remove_navigation_elements:
            data[
                "html_remove_navigation_elements"
            ] = self.html_remove_navigation_elements

        if self.http_proxy is not None:
            data["http_proxy"] = self.http_proxy

        if input_url is not None:
            files = None
            data["input_url"] = str(input_url)

        if input_s3_path is not None:
            files = None
            data["input_s3_path"] = str(input_s3_path)

        if self.invalidate_cache:
            data["invalidate_cache"] = self.invalidate_cache

        if self.is_formatting_instruction:
            data["is_formatting_instruction"] = self.is_formatting_instruction

        if self.language:
            data["language"] = self.language

        if self.max_pages is not None:
            data["max_pages"] = self.max_pages

        if self.output_pdf_of_document:
            data["output_pdf_of_document"] = self.output_pdf_of_document

        if self.output_s3_path_prefix is not None:
            data["output_s3_path_prefix"] = self.output_s3_path_prefix

        if self.page_prefix is not None:
            data["page_prefix"] = self.page_prefix

        # only send page separator to server if it is not None
        # as if a null, "" string is sent the server will then ignore the page separator instead of using the default
        if self.page_separator is not None:
            data["page_separator"] = self.page_separator

        if self.page_suffix is not None:
            data["page_suffix"] = self.page_suffix

        if self.parsing_instruction is not None:
            data["parsing_instruction"] = self.parsing_instruction

        if self.premium_mode:
            data["premium_mode"] = self.premium_mode

        if self.skip_diagonal_text:
            data["skip_diagonal_text"] = self.skip_diagonal_text

        if self.structured_output:
            data["structured_output"] = self.structured_output

        if self.structured_output_json_schema is not None:
            data["structured_output_json_schema"] = self.structured_output_json_schema

        if self.structured_output_json_schema_name is not None:
            data[
                "structured_output_json_schema_name"
            ] = self.structured_output_json_schema_name

        if self.take_screenshot:
            data["take_screenshot"] = self.take_screenshot

        if self.target_pages is not None:
            data["target_pages"] = self.target_pages

        if self.use_vendor_multimodal_model:
            data["use_vendor_multimodal_model"] = self.use_vendor_multimodal_model

        if self.vendor_multimodal_api_key is not None:
            data["vendor_multimodal_api_key"] = self.vendor_multimodal_api_key

        if self.vendor_multimodal_model_name is not None:
            data["vendor_multimodal_model_name"] = self.vendor_multimodal_model_name

        if self.webhook_url is not None:
            data["webhook_url"] = self.webhook_url

        # Deprecated
        if self.bounding_box is not None:
            data["bounding_box"] = self.bounding_box

        if self.gpt4o_mode:
            data["gpt4o_mode"] = self.gpt4o_mode

        if self.gpt4o_api_key is not None:
            data["gpt4o_api_key"] = self.gpt4o_api_key

        try:
            async with self.client_context() as client:
                response = await client.post(
                    url,
                    files=files,
                    headers=headers,
                    data=data,
                )
                if not response.is_success:
                    raise Exception(f"Failed to parse the file: {response.text}")
                job_id = response.json()["id"]
                return job_id
        finally:
            if file_handle is not None:
                file_handle.close()

    async def _get_job_result(
        self, job_id: str, result_type: str, verbose: bool = False
    ) -> Dict[str, Any]:
        result_url = f"{self.base_url}/api/parsing/job/{job_id}/result/{result_type}"
        status_url = f"{self.base_url}/api/parsing/job/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start = time.time()
        tries = 0
        while True:
            await asyncio.sleep(self.check_interval)
            async with self.client_context() as client:
                tries += 1

                result = await client.get(status_url, headers=headers)

                if result.status_code != 200:
                    end = time.time()
                    if end - start > self.max_timeout:
                        raise Exception(f"Timeout while parsing the file: {job_id}")
                    if verbose and tries % 10 == 0:
                        print(".", end="", flush=True)

                    await asyncio.sleep(self.check_interval)

                    continue

                # Allowed values "PENDING", "SUCCESS", "ERROR", "CANCELED"
                result_json = result.json()
                status = result_json["status"]
                if status == "SUCCESS":
                    parsed_result = await client.get(result_url, headers=headers)
                    return parsed_result.json()
                elif status == "PENDING":
                    end = time.time()
                    if end - start > self.max_timeout:
                        raise Exception(f"Timeout while parsing the file: {job_id}")
                    if verbose and tries % 10 == 0:
                        print(".", end="", flush=True)

                    await asyncio.sleep(self.check_interval)
                else:
                    error_code = result_json.get("error_code", "No error code found")
                    error_message = result_json.get(
                        "error_message", "No error message found"
                    )

                    exception_str = f"Job ID: {job_id} failed with status: {status}, Error code: {error_code}, Error message: {error_message}"
                    raise Exception(exception_str)

    async def _aload_data(
        self,
        file_path: FileInput,
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
        verbose: bool = False,
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info, fs=fs)
            if verbose:
                print("Started parsing the file under job_id %s" % job_id)

            result = await self._get_job_result(
                job_id, self.result_type.value, verbose=verbose
            )

            docs = [
                Document(
                    text=result[self.result_type.value],
                    metadata=extra_info or {},
                )
            ]
            if self.split_by_page:
                return self._get_sub_docs(docs)
            else:
                return docs

        except Exception as e:
            file_repr = file_path if isinstance(file_path, str) else "<bytes/buffer>"
            print(f"Error while parsing the file '{file_repr}':", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    async def aload_data(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, PurePosixPath, Path, bytes, BufferedIOBase)):
            return await self._aload_data(
                file_path, extra_info=extra_info, fs=fs, verbose=self.verbose
            )
        elif isinstance(file_path, list):
            jobs = [
                self._aload_data(
                    f,
                    extra_info=extra_info,
                    fs=fs,
                    verbose=self.verbose and not self.show_progress,
                )
                for f in file_path
            ]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )

                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise e
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def load_data(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            return asyncio_run(self.aload_data(file_path, extra_info, fs=fs))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def _aget_json(
        self, file_path: FileInput, extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info)
            if self.verbose:
                print("Started parsing the file under job_id %s" % job_id)
            result = await self._get_job_result(job_id, "json")
            result["job_id"] = job_id

            if not isinstance(file_path, (bytes, BufferedIOBase)):
                result["file_path"] = str(file_path)

            return [result]
        except Exception as e:
            file_repr = file_path if isinstance(file_path, str) else "<bytes/buffer>"
            print(f"Error while parsing the file '{file_repr}':", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    async def aget_json(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
    ) -> List[dict]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            return await self._aget_json(file_path, extra_info=extra_info)
        elif isinstance(file_path, list):
            jobs = [self._aget_json(f, extra_info=extra_info) for f in file_path]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )

                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise e
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def get_json_result(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
    ) -> List[dict]:
        """Parse the input path."""
        try:
            return asyncio_run(self.aget_json(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aget_images(
        self, json_result: List[dict], download_path: str
    ) -> List[dict]:
        """Download images from the parsed result."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # make the download path
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        try:
            images = []
            for result in json_result:
                job_id = result["job_id"]
                for page in result["pages"]:
                    if self.verbose:
                        print(f"> Image for page {page['page']}: {page['images']}")
                    for image in page["images"]:
                        image_name = image["name"]

                        # get the full path
                        image_path = os.path.join(
                            download_path, f"{job_id}-{image_name}"
                        )

                        # get a valid image path
                        if not image_path.endswith(".png"):
                            if not image_path.endswith(".jpg"):
                                image_path += ".png"

                        image["path"] = image_path
                        image["job_id"] = job_id

                        image["original_file_path"] = result.get("file_path", None)

                        image["page_number"] = page["page"]
                        with open(image_path, "wb") as f:
                            image_url = f"{self.base_url}/api/parsing/job/{job_id}/result/image/{image_name}"
                            async with self.client_context() as client:
                                res = await client.get(
                                    image_url, headers=headers, timeout=self.max_timeout
                                )
                                res.raise_for_status()
                                f.write(res.content)
                        images.append(image)
            return images
        except Exception as e:
            print("Error while downloading images from the parsed result:", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    def get_images(self, json_result: List[dict], download_path: str) -> List[dict]:
        """Download images from the parsed result."""
        try:
            return asyncio_run(self.aget_images(json_result, download_path))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aget_xlsx(
        self, json_result: List[dict], download_path: str
    ) -> List[dict]:
        """Download images from the parsed result."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # make the download path
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        try:
            xlsx_list = []
            for result in json_result:
                job_id = result["job_id"]
                if self.verbose:
                    print("> XLSX")

                xlsx_path = os.path.join(download_path, f"{job_id}.xlsx")

                xlsx = {}

                xlsx["path"] = xlsx_path
                xlsx["job_id"] = job_id
                xlsx["original_file_path"] = result.get("file_path", None)

                with open(xlsx_path, "wb") as f:
                    xlsx_url = (
                        f"{self.base_url}/api/parsing/job/{job_id}/result/raw/xlsx"
                    )
                    async with self.client_context() as client:
                        res = await client.get(
                            xlsx_url, headers=headers, timeout=self.max_timeout
                        )
                        res.raise_for_status()
                        f.write(res.content)
                xlsx_list.append(xlsx)
            return xlsx_list

        except Exception as e:
            print("Error while downloading xlsx:", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    def get_xlsx(self, json_result: List[dict], download_path: str) -> List[dict]:
        """Download xlsx from the parsed result."""
        try:
            return asyncio_run(self.aget_xlsx(json_result, download_path))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    def _get_sub_docs(self, docs: List[Document]) -> List[Document]:
        """Split docs into pages, by separator."""
        sub_docs = []
        separator = self.page_separator or _DEFAULT_SEPARATOR
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                sub_doc = Document(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                sub_docs.append(sub_doc)

        return sub_docs
