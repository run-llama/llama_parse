import os
import pytest
import uuid
from typing import AsyncGenerator
from pytest_asyncio import fixture as async_fixture
from llama_cloud_services.report import LlamaReport, ReportClient

# Skip tests if no API key is set
pytestmark = pytest.mark.skipif(
    not os.getenv("LLAMA_CLOUD_API_KEY"), reason="No API key provided"
)


@async_fixture(scope="function")
async def client() -> AsyncGenerator[LlamaReport, None]:
    """Create a LlamaReport client."""
    client = LlamaReport()
    reports_before = await client.alist_reports()
    reports_before_ids = [r.report_id for r in reports_before]
    try:
        yield client
    finally:
        # clean up reports
        try:
            reports_after = await client.alist_reports()
            reports_after_ids = [r.report_id for r in reports_after]
            for report_id in reports_before_ids:
                if report_id not in reports_after_ids:
                    await client.adelete_report(report_id)
        finally:
            await client.aclient.aclose()


@pytest.fixture(scope="function")
def unique_name() -> str:
    """Generate a unique report name."""
    return f"test-report-{uuid.uuid4()}"


@async_fixture(scope="function")
async def report(
    client: LlamaReport, unique_name: str
) -> AsyncGenerator[ReportClient, None]:
    """Create a report."""
    report = await client.acreate_report(
        name=unique_name,
        template_text=(
            "# [Some title]\n\n"
            " ## TLDR\n"
            "A quick summary of the paper.\n\n"
            "## Details\n"
            "More details about the paper, possible more than one section here.\n"
        ),
        input_files=["tests/test_files/paper.md"],
    )
    try:
        yield report
    finally:
        await report.adelete()


@pytest.mark.asyncio
async def test_create_and_delete_report(
    client: LlamaReport, report: ReportClient
) -> None:
    """Test basic report creation and deletion."""
    # Verify the report exists
    metadata = await report.aget_metadata()
    assert metadata.name == report.name

    # Test listing reports
    reports = await client.alist_reports()
    assert any(r.report_id == report.report_id for r in reports)

    # Test getting report by ID
    fetched_report = await client.aget_report(report.report_id)
    assert fetched_report.report_id == report.report_id
    assert fetched_report.name == report.name


@pytest.mark.asyncio
async def test_report_plan_workflow(report: ReportClient) -> None:
    """Test the report planning workflow."""
    # Wait for the plan
    plan = await report.await_for_plan()
    assert plan is not None

    # Approve the plan
    response = await report.aupdate_plan(action="approve")
    assert response is not None

    # Wait for completion
    completed_report = await report.await_completion()
    assert len(completed_report.blocks) > 0

    # Get edit suggestions
    suggestions = await report.asuggest_edits(
        "Make the text more formal.", auto_history=True
    )
    assert len(suggestions) > 0

    # Test accepting an edit
    await report.aaccept_edit(suggestions[0])

    # Get more suggestions and test rejecting
    more_suggestions = await report.asuggest_edits(
        "Add a section about machine learning.", auto_history=True
    )
    assert len(more_suggestions) > 0
    await report.areject_edit(more_suggestions[0])

    # Verify chat history is maintained
    assert len(report.chat_history) >= 4  # 2 user messages + 2 assistant responses

    # get events
    events = await report.aget_events()
    assert len(events) > 0
