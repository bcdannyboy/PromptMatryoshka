#!/usr/bin/env python3
"""Comprehensive integration test validation script for PromptMatryoshka.

This script validates the complete multi-provider integration test suite and provides
detailed reporting on test coverage, results, and findings.

Integration Test Coverage:
- Multi-provider system integration (4 providers: OpenAI, Anthropic, Ollama, HuggingFace)
- Configuration profiles integration (6 profiles: research-openai, production-anthropic, etc.)
- CLI commands integration (8 new commands: list-providers, check-provider, etc.)
- Plugin system integration with multi-provider support
- Configuration validation and error handling
- Pipeline integration with provider switching
- Error handling and edge cases
- Performance and reliability characteristics

Usage:
    python3 test_integration_validation.py [--verbose] [--report-only]
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse


@dataclass
class TestResult:
    """Test result data structure."""
    test_file: str
    test_count: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    success_rate: float
    details: Dict[str, Any]


@dataclass
class IntegrationTestSuite:
    """Integration test suite metadata."""
    name: str
    file_path: str
    description: str
    test_count_estimate: int
    coverage_areas: List[str]
    priority: int  # 1=critical, 2=important, 3=nice-to-have


class TestValidator:
    """Validates integration test coverage and results."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.integration_test_suites = self._define_integration_test_suites()
        self.results: List[TestResult] = []
        
    def _define_integration_test_suites(self) -> List[IntegrationTestSuite]:
        """Define all integration test suites created."""
        return [
            IntegrationTestSuite(
                name="Multi-Provider Integration",
                file_path="tests/test_multi_provider_integration.py",
                description="Tests multi-provider LLM system integration across 4 providers",
                test_count_estimate=15,
                coverage_areas=[
                    "Provider discovery and availability",
                    "Configuration profile creation (6 profiles)",
                    "Provider switching and fallback",
                    "Plugin integration with providers",
                    "Error handling and recovery",
                    "Concurrent operations",
                    "Performance and caching",
                    "Environment variable resolution"
                ],
                priority=1
            ),
            IntegrationTestSuite(
                name="CLI Integration", 
                file_path="tests/test_cli_integration.py",
                description="Tests all 8 new CLI commands with full integration",
                test_count_estimate=25,
                coverage_areas=[
                    "list-providers command",
                    "check-provider command", 
                    "test-provider command",
                    "list-profiles command",
                    "show-profile command",
                    "validate-config command",
                    "show-config command",
                    "config-health command",
                    "Enhanced run command with --provider/--profile"
                ],
                priority=1
            ),
            IntegrationTestSuite(
                name="Configuration Profiles Integration",
                file_path="tests/test_configuration_profiles_integration.py", 
                description="Tests all 6 configuration profiles with provider integration",
                test_count_estimate=20,
                coverage_areas=[
                    "research-openai profile",
                    "production-anthropic profile",
                    "local-development profile", 
                    "fast-gpt35 profile",
                    "creative-anthropic profile",
                    "local-llama profile",
                    "Parameter inheritance and overrides",
                    "Profile switching runtime behavior"
                ],
                priority=1
            ),
            IntegrationTestSuite(
                name="Plugin System Integration",
                file_path="tests/test_plugin_system_integration.py",
                description="Tests plugin system integration with multi-provider support",
                test_count_estimate=22,
                coverage_areas=[
                    "Plugin discovery and registration",
                    "Mixed plugin pipeline execution", 
                    "LLM-based plugin provider integration",
                    "Technique-based plugin compatibility",
                    "Plugin configuration inheritance",
                    "Error isolation and recovery",
                    "Performance characteristics"
                ],
                priority=1
            ),
            IntegrationTestSuite(
                name="Configuration Validation Integration",
                file_path="tests/test_configuration_validation_integration.py",
                description="Tests configuration system validation and error handling",
                test_count_estimate=18,
                coverage_areas=[
                    "Pydantic v2 schema validation",
                    "Configuration error messages",
                    "Environment variable resolution",
                    "Legacy configuration conversion",
                    "Validation edge cases",
                    "Provider configuration validation"
                ],
                priority=2
            ),
            IntegrationTestSuite(
                name="Pipeline Integration",
                file_path="tests/test_pipeline_integration.py",
                description="Tests pipeline integration with multi-provider support", 
                test_count_estimate=25,
                coverage_areas=[
                    "Basic pipeline execution across providers",
                    "Provider switching during runtime",
                    "Profile-based execution",
                    "Mixed plugin execution",
                    "Error handling and recovery",
                    "Performance characteristics",
                    "Dependency resolution",
                    "Complex workflows"
                ],
                priority=1
            ),
            IntegrationTestSuite(
                name="Error Handling Integration",
                file_path="tests/test_error_handling_integration.py",
                description="Tests comprehensive error handling and edge cases",
                test_count_estimate=30,
                coverage_areas=[
                    "Provider connection failures",
                    "Configuration validation errors",
                    "Plugin error isolation",
                    "Resource exhaustion scenarios",
                    "Invalid input handling",
                    "Concurrent error scenarios",
                    "Recovery mechanisms",
                    "Error message quality"
                ],
                priority=2
            ),
            IntegrationTestSuite(
                name="Performance & Reliability Integration",
                file_path="tests/test_performance_reliability_integration.py",
                description="Tests performance and reliability characteristics",
                test_count_estimate=20,
                coverage_areas=[
                    "Performance benchmarking across providers",
                    "Load testing scenarios",
                    "Memory usage patterns",
                    "Concurrent execution performance",
                    "Caching effectiveness",
                    "Scalability testing",
                    "Resource consumption monitoring",
                    "Reliability under sustained load"
                ],
                priority=2
            )
        ]
    
    def validate_test_files_exist(self) -> Dict[str, bool]:
        """Validate that all integration test files exist."""
        existence_map = {}
        for suite in self.integration_test_suites:
            file_path = self.project_root / suite.file_path
            exists = file_path.exists()
            existence_map[suite.name] = exists
            if self.verbose:
                status = "✓" if exists else "✗"
                print(f"{status} {suite.name}: {suite.file_path}")
        return existence_map
    
    def run_test_suite(self, suite: IntegrationTestSuite) -> TestResult:
        """Run a specific test suite and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {suite.name}")
        print(f"File: {suite.file_path}")
        print(f"Description: {suite.description}")
        print(f"{'='*60}")
        
        file_path = self.project_root / suite.file_path
        if not file_path.exists():
            return TestResult(
                test_file=suite.file_path,
                test_count=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=0.0,
                success_rate=0.0,
                details={"error": "Test file does not exist"}
            )
        
        # Run pytest with JSON output
        start_time = time.time()
        cmd = [
            sys.executable, "-m", "pytest",
            str(file_path),
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=/tmp/pytest_report.json"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            duration = time.time() - start_time
            
            # Parse pytest output
            test_count = 0
            passed = 0
            failed = 0
            errors = 0
            skipped = 0
            
            # Try to parse JSON report if available
            json_report_path = Path("/tmp/pytest_report.json")
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        report_data = json.load(f)
                    
                    summary = report_data.get('summary', {})
                    test_count = summary.get('total', 0)
                    passed = summary.get('passed', 0)
                    failed = summary.get('failed', 0)
                    errors = summary.get('error', 0)
                    skipped = summary.get('skipped', 0)
                    
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Fallback: parse text output
            if test_count == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line and ('failed' in line or 'error' in line):
                        # Parse summary line like "2 passed, 13 errors in 0.88s"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed' and i > 0:
                                passed = int(parts[i-1])
                            elif part == 'failed' and i > 0:
                                failed = int(parts[i-1])
                            elif part in ['error', 'errors'] and i > 0:
                                errors = int(parts[i-1])
                            elif part == 'skipped' and i > 0:
                                skipped = int(parts[i-1])
                
                test_count = passed + failed + errors + skipped
            
            success_rate = (passed / max(1, test_count)) * 100
            
            # Print results
            print(f"Results: {test_count} tests, {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Duration: {duration:.2f}s")
            
            if self.verbose and (failed > 0 or errors > 0):
                print(f"\nTest Output:\n{result.stdout}")
                if result.stderr:
                    print(f"\nErrors:\n{result.stderr}")
            
            return TestResult(
                test_file=suite.file_path,
                test_count=test_count,
                passed=passed,
                failed=failed,
                errors=errors,
                skipped=skipped,
                duration=duration,
                success_rate=success_rate,
                details={
                    "exit_code": result.returncode,
                    "stdout_lines": len(result.stdout.split('\n')),
                    "stderr_lines": len(result.stderr.split('\n')) if result.stderr else 0
                }
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"❌ Test suite timed out after {duration:.1f}s")
            return TestResult(
                test_file=suite.file_path,
                test_count=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=duration,
                success_rate=0.0,
                details={"error": "Test suite timed out"}
            )
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error running test suite: {e}")
            return TestResult(
                test_file=suite.file_path,
                test_count=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=duration,
                success_rate=0.0,
                details={"error": str(e)}
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all integration test suites."""
        print("Starting Integration Test Validation")
        print(f"Project Root: {self.project_root}")
        print(f"Test Directory: {self.test_dir}")
        print(f"Total Test Suites: {len(self.integration_test_suites)}")
        
        # Validate test files exist
        print(f"\n{'='*60}")
        print("VALIDATING TEST FILE EXISTENCE")
        print(f"{'='*60}")
        existence_map = self.validate_test_files_exist()
        existing_files = sum(existence_map.values())
        print(f"\nTest Files: {existing_files}/{len(self.integration_test_suites)} exist")
        
        # Run each test suite
        self.results = []
        for suite in self.integration_test_suites:
            if existence_map.get(suite.name, False):
                result = self.run_test_suite(suite)
                self.results.append(result)
            else:
                print(f"\n❌ Skipping {suite.name} - file does not exist")
                self.results.append(TestResult(
                    test_file=suite.file_path,
                    test_count=0,
                    passed=0,
                    failed=0,
                    errors=1,
                    skipped=0,
                    duration=0.0,
                    success_rate=0.0,
                    details={"error": "File not found"}
                ))
        
        return self.results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test validation report."""
        total_tests = sum(r.test_count for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_duration = sum(r.duration for r in self.results)
        
        overall_success_rate = (total_passed / max(1, total_tests)) * 100
        
        # Coverage analysis
        coverage_areas = set()
        for suite in self.integration_test_suites:
            coverage_areas.update(suite.coverage_areas)
        
        # Priority analysis
        critical_suites = [s for s in self.integration_test_suites if s.priority == 1]
        critical_results = [r for r, s in zip(self.results, self.integration_test_suites) if s.priority == 1]
        critical_success_rate = sum(r.passed for r in critical_results) / max(1, sum(r.test_count for r in critical_results)) * 100
        
        return {
            "summary": {
                "total_test_suites": len(self.integration_test_suites),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "total_skipped": total_skipped,
                "overall_success_rate": overall_success_rate,
                "total_duration": total_duration,
                "critical_success_rate": critical_success_rate
            },
            "coverage": {
                "total_coverage_areas": len(coverage_areas),
                "coverage_areas": sorted(list(coverage_areas)),
                "critical_suites": len(critical_suites),
                "provider_coverage": ["OpenAI", "Anthropic", "Ollama", "HuggingFace"],
                "profile_coverage": [
                    "research-openai", "production-anthropic", "local-development",
                    "fast-gpt35", "creative-anthropic", "local-llama"
                ],
                "cli_command_coverage": [
                    "list-providers", "check-provider", "test-provider",
                    "list-profiles", "show-profile", "validate-config",
                    "show-config", "config-health"
                ]
            },
            "detailed_results": [
                {
                    "suite_name": suite.name,
                    "file_path": result.test_file,
                    "test_count": result.test_count,
                    "passed": result.passed,
                    "failed": result.failed,
                    "errors": result.errors,
                    "skipped": result.skipped,
                    "success_rate": result.success_rate,
                    "duration": result.duration,
                    "priority": suite.priority,
                    "coverage_areas": suite.coverage_areas,
                    "details": result.details
                }
                for suite, result in zip(self.integration_test_suites, self.results)
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed critical tests
        critical_failures = [
            (suite, result) for suite, result in zip(self.integration_test_suites, self.results)
            if suite.priority == 1 and (result.failed > 0 or result.errors > 0)
        ]
        
        if critical_failures:
            recommendations.append(
                f"🔥 CRITICAL: {len(critical_failures)} critical test suites have failures. "
                "These should be addressed immediately as they test core multi-provider functionality."
            )
        
        # Check overall success rate
        total_tests = sum(r.test_count for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        success_rate = (total_passed / max(1, total_tests)) * 100
        
        if success_rate < 80:
            recommendations.append(
                f"⚠️  Overall success rate is {success_rate:.1f}%. "
                "Consider investigating and fixing failing tests before deployment."
            )
        elif success_rate < 95:
            recommendations.append(
                f"⚡ Overall success rate is {success_rate:.1f}%. "
                "Good progress, but consider addressing remaining failures."
            )
        else:
            recommendations.append(
                f"✅ Excellent overall success rate of {success_rate:.1f}%. "
                "Integration tests are passing well."
            )
        
        # Check for missing test files
        missing_files = [
            suite.name for suite, result in zip(self.integration_test_suites, self.results)
            if "File not found" in result.details.get("error", "")
        ]
        
        if missing_files:
            recommendations.append(
                f"📁 {len(missing_files)} test suites have missing files: {', '.join(missing_files[:3])}{'...' if len(missing_files) > 3 else ''}. "
                "These test files should be created to complete coverage."
            )
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print comprehensive test report."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE INTEGRATION TEST VALIDATION REPORT")
        print(f"{'='*80}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary
        summary = report["summary"]
        print(f"\n📊 SUMMARY")
        print(f"{'─'*40}")
        print(f"Test Suites: {summary['total_test_suites']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']} ✅")
        print(f"Failed: {summary['total_failed']} ❌")
        print(f"Errors: {summary['total_errors']} 🔥")
        print(f"Skipped: {summary['total_skipped']} ⏭️")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Critical Success Rate: {summary['critical_success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.1f}s")
        
        # Coverage
        coverage = report["coverage"]
        print(f"\n🎯 COVERAGE ANALYSIS")
        print(f"{'─'*40}")
        print(f"Coverage Areas: {coverage['total_coverage_areas']}")
        print(f"Providers: {', '.join(coverage['provider_coverage'])}")
        print(f"Profiles: {len(coverage['profile_coverage'])} configuration profiles")
        print(f"CLI Commands: {len(coverage['cli_command_coverage'])} new CLI commands")
        
        # Detailed Results
        print(f"\n📋 DETAILED RESULTS")
        print(f"{'─'*80}")
        for result in report["detailed_results"]:
            priority_icon = "🔥" if result["priority"] == 1 else "⚡" if result["priority"] == 2 else "💡"
            status_icon = "✅" if result["success_rate"] >= 95 else "⚠️" if result["success_rate"] >= 80 else "❌"
            
            print(f"{priority_icon}{status_icon} {result['suite_name']}")
            print(f"    File: {result['file_path']}")
            print(f"    Tests: {result['test_count']} | Passed: {result['passed']} | Failed: {result['failed']} | Errors: {result['errors']}")
            print(f"    Success Rate: {result['success_rate']:.1f}% | Duration: {result['duration']:.2f}s")
            
            if result['failed'] > 0 or result['errors'] > 0:
                print(f"    ⚠️  Has failures/errors - requires attention")
            
            print()
        
        # Recommendations
        print(f"🔧 RECOMMENDATIONS")
        print(f"{'─'*40}")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print(f"\n{'='*80}")
        print("INTEGRATION TEST VALIDATION COMPLETE")
        print(f"{'='*80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate PromptMatryoshka integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report-only", "-r", action="store_true", help="Generate report without running tests")
    parser.add_argument("--save-report", "-s", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    validator = TestValidator(verbose=args.verbose)
    
    if not args.report_only:
        # Run all tests
        results = validator.run_all_tests()
    
    # Generate and print report
    report = validator.generate_comprehensive_report()
    validator.print_report(report)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {args.save_report}")
    
    # Exit with appropriate code
    summary = report["summary"]
    if summary["total_errors"] > 0:
        sys.exit(2)  # Critical errors
    elif summary["total_failed"] > 0:
        sys.exit(1)  # Test failures
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()