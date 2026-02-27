"""Tests for MCP tool functions."""

from dirac_mcp.tools.jobs import create_basic_jdl


class TestCreateBasicJdl:
    """Tests for the create_basic_jdl pure function."""

    def test_minimal_jdl(self):
        result = create_basic_jdl(executable="/bin/echo")
        assert 'Executable = "/bin/echo";' in result
        assert 'JobName = "Auto-generated Job";' in result
        assert 'StdOutput = "StdOut";' in result
        assert 'StdError = "StdErr";' in result
        assert 'OutputSandbox = {"StdOut", "StdErr"};' in result

    def test_custom_job_name(self):
        result = create_basic_jdl(executable="/bin/ls", job_name="My Job")
        assert 'JobName = "My Job";' in result

    def test_with_arguments(self):
        result = create_basic_jdl(executable="/bin/echo", arguments="hello world")
        assert 'Arguments = "hello world";' in result

    def test_without_arguments(self):
        result = create_basic_jdl(executable="/bin/echo")
        assert "Arguments" not in result

    def test_with_input_sandbox(self):
        result = create_basic_jdl(executable="/bin/run.sh", input_sandbox=["run.sh", "data.txt"])
        assert 'InputSandbox = {"run.sh", "data.txt"};' in result

    def test_with_output_sandbox(self):
        result = create_basic_jdl(executable="/bin/echo", output_sandbox=["result.txt", "StdOut"])
        assert 'OutputSandbox = {"result.txt", "StdOut"};' in result

    def test_with_site(self):
        result = create_basic_jdl(executable="/bin/echo", site="LCG.CERN.ch")
        assert 'Site = "LCG.CERN.ch";' in result

    def test_without_site(self):
        result = create_basic_jdl(executable="/bin/echo")
        assert "Site" not in result

    def test_with_memory(self):
        result = create_basic_jdl(executable="/bin/echo", memory=2048)
        assert "Memory = 2048;" in result

    def test_with_max_cpu_time(self):
        result = create_basic_jdl(executable="/bin/echo", max_cpu_time=3600)
        assert "MaxCPUTime = 3600;" in result

    def test_full_jdl(self):
        result = create_basic_jdl(
            executable="/bin/run.sh",
            job_name="Full Test Job",
            arguments="--verbose",
            input_sandbox=["run.sh"],
            output_sandbox=["output.dat", "StdOut", "StdErr"],
            site="LCG.CERN.ch",
            memory=4096,
            max_cpu_time=7200,
        )
        assert 'JobName = "Full Test Job";' in result
        assert 'Executable = "/bin/run.sh";' in result
        assert 'Arguments = "--verbose";' in result
        assert 'InputSandbox = {"run.sh"};' in result
        assert 'Site = "LCG.CERN.ch";' in result
        assert "Memory = 4096;" in result
        assert "MaxCPUTime = 7200;" in result
