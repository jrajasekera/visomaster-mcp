"""Exercise actual stdio framing and schemas without starting a GPU job."""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_discovery():
    async def check():
        params = StdioServerParameters(command=sys.executable, args=["-m", "visomaster_mcp.server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                assert len(tools) == 24
                assert tools["health_check"].annotations.readOnlyHint
                normalize = tools["normalize_media"]
                assert not normalize.annotations.readOnlyHint
                assert not normalize.annotations.destructiveHint
                assert normalize.inputSchema["required"] == ["media_path"]
                for name in ("delete_job", "prune_storage"):
                    assert tools[name].annotations.destructiveHint
                prune = tools["prune_storage"].inputSchema["properties"]
                assert prune["dry_run"]["default"] is True
                assert prune["older_than_days"]["minimum"] == 0
                detect = tools["detect_faces"].inputSchema["properties"]
                assert {"frame", "sample_frames", "frames", "similarity_threshold"} <= detect.keys()
                assert detect["sample_frames"]["anyOf"][0]["maximum"] == 64
                for name in ("detect_faces", "submit_swap", "compare_variants"):
                    priority = tools[name].inputSchema["properties"]["priority"]
                    assert priority["anyOf"][0]["enum"] == ["interactive", "normal"]
                    assert priority["default"] is None
                    assert "priority" not in tools[name].inputSchema.get("required", [])
                assert "priority" not in tools["normalize_media"].inputSchema["properties"]
                assert tools["submit_swap"].outputSchema["type"] == "object"
                analyze = tools["analyze_references"]
                assert not analyze.annotations.readOnlyHint
                assert analyze.inputSchema["required"] == ["references"]
                assert analyze.inputSchema["properties"]["references"]["maxItems"] == 20
                assert analyze.outputSchema["type"] == "object"
                mapping = tools["submit_swap"].inputSchema["$defs"]["FaceMapping"]["properties"]
                assert {"source_weights", "source_sha256"} <= mapping.keys()
                assert "mappings" in tools["submit_swap"].inputSchema["required"]
                swap = tools["submit_swap"].inputSchema
                assert "report" in swap["properties"] and "report" not in swap["required"]
                # Preflight takes submit_swap's arguments, with identical schemas, plus one.
                preflight = tools["preflight_swap"]
                checked = preflight.inputSchema
                assert not preflight.annotations.readOnlyHint
                assert not preflight.annotations.destructiveHint
                assert set(checked["properties"]) == set(swap["properties"]) | {
                    "analyze_references"
                }
                for name, schema in swap["properties"].items():
                    assert checked["properties"][name] == schema, name
                assert checked["required"] == swap["required"]
                assert checked["$defs"] == swap["$defs"]
                assert checked["properties"]["analyze_references"]["default"] is False
                assert preflight.outputSchema["type"] == "object"
                options = swap["$defs"]["SwapReportOptions"]["properties"]
                assert {"review_images", "expected_visible", "min_coverage"} == options.keys()
                assert options["review_images"]["maximum"] == 24
                for name in ("save_recipe", "run_recipe", "export_recipe", "import_recipe"):
                    assert not tools[name].annotations.readOnlyHint
                    assert not tools[name].annotations.destructiveHint
                assert tools["get_recipe"].annotations.readOnlyHint
                assert tools["delete_recipe"].annotations.destructiveHint
                run = tools["run_recipe"].inputSchema
                assert run["required"] == ["name"]
                assert run["properties"]["accept_changes"]["default"] is False
                assert {"media_path", "targets", "frame_range", "report", "priority"} <= (
                    run["properties"].keys()
                )
                assert tools["save_recipe"].inputSchema["required"] == ["name"]
                assert tools["import_recipe"].inputSchema["required"] == ["local_path"]
                compare = tools["compare_variants"]
                assert not compare.annotations.readOnlyHint
                assert compare.inputSchema["required"] == ["media_path", "mappings", "variants"]
                schema = compare.inputSchema["properties"]
                assert (schema["variants"]["minItems"], schema["variants"]["maxItems"]) == (2, 6)
                assert schema["frames"]["anyOf"][0]["maxItems"] == 4
                assert schema["max_frames"]["anyOf"][0]["maximum"] == 120
                variant = compare.inputSchema["$defs"]["Variant"]
                assert variant["properties"].keys() == {"name", "parameters", "controls"}
                assert variant["additionalProperties"] is False
                assert compare.outputSchema["type"] == "object"
                resume = tools["resume_render"]
                assert not resume.annotations.readOnlyHint
                assert not resume.annotations.destructiveHint
                assert resume.inputSchema["required"] == ["from_job_id"]
                assert resume.inputSchema["properties"].keys() == {
                    "from_job_id",
                    "job_id",
                    "priority",
                }
                result = await session.call_tool(
                    "submit_swap", {"media_path": "x", "mappings": [{}]}
                )
                assert result.isError

    asyncio.run(check())


def test_health_compares_deployed_code_with_checkout(tmp_path, monkeypatch):
    import hashlib

    from visomaster_mcp import server

    deployed = {
        name: hashlib.sha256((server.LOCAL_CODE / name).read_bytes()).hexdigest()
        for name in ("common.py", "control.py", "engine.py")
    }

    async def health(code):
        async def command(action):
            return {"release": "r1", "code": code} if code else {"revision": "x"}

        monkeypatch.setattr(server, "command", command)
        return (await server.health_check())["code_matches"]

    assert asyncio.run(health(deployed)) is True
    assert asyncio.run(health({**deployed, "engine.py": "0" * 64})) is False
    assert asyncio.run(health(None)) is None
    monkeypatch.setattr(server, "LOCAL_CODE", tmp_path / "missing")
    assert asyncio.run(health(deployed)) is None
