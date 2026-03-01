"""Tests for common.py -- language detection and chunk splitting."""

from common import detect_language, split_into_chunks, _find_cpp_boundaries, _is_cpp_comment_line


# -- detect_language ----------------------------------------------------------

def test_detect_language_python():
    ctx = """
import os
from pathlib import Path

class Config:
    pass

class Runner(ABC):
    def run(self):
        pass

import numpy as np
from torch import nn
""".strip()
    assert detect_language(ctx) == "py"


def test_detect_language_cpp():
    ctx = """
#include <types.h>
#include <system.h>

class Binning {
protected:
    System* system;
};

struct IPHeader {
    int version;
};

#include <iostream>
#include <vector>
""".strip()
    assert detect_language(ctx) == "cpp"


def test_detect_language_ts():
    ctx = """
import { Component } from '@angular/core';
import { Router } from '@angular/router';

interface FileShareProps {
    name: string;
}

interface FileListProps {
    files: File[];
}

export class AppComponent {
    title = 'app';
}
""".strip()
    assert detect_language(ctx) == "ts"


def test_detect_language_unknown():
    ctx = "just some plain text\nwith no code signals\n"
    assert detect_language(ctx) == "unknown"


# -- _is_cpp_comment_line ----------------------------------------------------

def test_cpp_comment_line_slash():
    assert _is_cpp_comment_line("// this is a comment")
    assert _is_cpp_comment_line("//****************")


def test_cpp_comment_line_block():
    assert _is_cpp_comment_line("/* start of block */")
    assert _is_cpp_comment_line(" * middle of block")
    assert _is_cpp_comment_line(" */")


def test_cpp_comment_line_not_comment():
    assert not _is_cpp_comment_line("#include <foo.h>")
    assert not _is_cpp_comment_line("int main() {")
    assert not _is_cpp_comment_line("")


# -- _find_cpp_boundaries ----------------------------------------------------

def test_find_cpp_boundaries_slash_comment():
    """Comment block with // followed by blank + #include."""
    lines = [
        "int x = 0;",           # 0
        "",                      # 1
        "//****************",    # 2 - comment block start
        "//  ExaMiniMD v1.0",   # 3
        "//  Copyright 2018",   # 4
        "//****************",    # 5 - comment block end
        "",                      # 6
        "#include <examinimd.h>",# 7
        "#include <iostream>",   # 8
        "int main() {",          # 9
    ]
    boundaries = _find_cpp_boundaries(lines)
    assert 2 in boundaries


def test_find_cpp_boundaries_block_comment():
    """C-style /* */ comment block followed by #include."""
    lines = [
        "something();",          # 0
        "",                      # 1
        "/*!",                   # 2
        " * \\file",            # 3
        " * \\brief Graphs",    # 4
        " */",                   # 5
        "",                      # 6
        "#include <GKlib.h>",    # 7
        "",                      # 8
    ]
    boundaries = _find_cpp_boundaries(lines)
    assert 2 in boundaries


def test_find_cpp_boundaries_no_blank_before_comment():
    """No boundary when comment block is not preceded by blank line."""
    lines = [
        "int x = 0;",           # 0 - non-blank, directly before comment
        "// Copyright",          # 1
        "//****************",    # 2
        "",                      # 3
        "#include <foo.h>",      # 4
    ]
    boundaries = _find_cpp_boundaries(lines)
    assert boundaries == []


def test_find_cpp_boundaries_include_without_comment():
    """No boundary when #include appears without preceding comment block."""
    lines = [
        "int x = 0;",           # 0
        "",                      # 1
        "#include <foo.h>",      # 2
        "#include <bar.h>",      # 3
    ]
    boundaries = _find_cpp_boundaries(lines)
    assert boundaries == []


# -- split_into_chunks for C++ -----------------------------------------------

def test_split_cpp_two_files():
    """Two C++ files separated by comment block + #include."""
    ctx = "\n".join([
        "//****************",
        "//  File 1",
        "//****************",
        "",
        "#include <types.h>",
        "#include <system.h>",
        "class Binning {",
        "protected:",
        "    System* system;",
        "};",
        "",
        "void Binning::init() {",
        "    // init code",
        "}",
        "",
        "void Binning::update() {",
        "    // update code",
        "}",
        "",
        "void Binning::cleanup() {",
        "    // cleanup",
        "}",
        "",
        "",
        "//****************",
        "//  File 2",
        "//****************",
        "",
        "#include <examinimd.h>",
        "#include <iostream>",
        "int main() {",
        "    return 0;",
        "}",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2
    assert chunks[0].filename_hint == "types.h"
    assert chunks[1].filename_hint == "examinimd.h"


def test_split_cpp_c_style_comment():
    """C-style block comment before #include triggers boundary."""
    ctx = "\n".join([
        "/*",
        " * First file",
        " */",
        "",
        "#include <GKlib.h>",
        "",
        "void graph_create() {",
        "    // code",
        "}",
        "",
        "void graph_free() {",
        "    // code",
        "}",
        "",
        "void graph_read() {",
        "    // code",
        "}",
        "",
        "void graph_write() {",
        "    // code",
        "}",
        "",
        "",
        "/*",
        " * Second file",
        " * by Author",
        " */",
        "",
        "#include <GKlib.h>",
        "",
        "void seq_init() {",
        "    // code",
        "}",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2
    assert chunks[0].filename_hint == "GKlib.h"
    assert chunks[1].filename_hint == "GKlib.h"


def test_split_cpp_no_hint_for_cmake():
    """CMake chunk should not get a filename_hint."""
    ctx = "\n".join([
        "cmake_minimum_required(VERSION 3.13)",
        "project(foo)",
        "set(SOURCES main.cpp)",
        "",
        "add_executable(foo ${SOURCES})",
        "",
        "# some more cmake stuff",
        "target_link_libraries(foo bar)",
        "",
        "# even more stuff",
        "install(TARGETS foo)",
        "",
        "# and more stuff",
        "set(FLAGS -Wall)",
        "",
        "# final cmake",
        "message(STATUS done)",
        "",
        "# end of cmake",
        "set(X 1)",
        "",
        "",
        "//****************",
        "//  Source file",
        "//****************",
        "",
        "#include <main.h>",
        "int main() { return 0; }",
    ])
    chunks = split_into_chunks(ctx)
    assert chunks[0].filename_hint is None
    assert any(ch.filename_hint == "main.h" for ch in chunks)


def test_split_cpp_preserves_python_splitting():
    """Python context should still split normally with no C++ boundaries."""
    ctx = "\n".join([
        '"""Module docstring."""',
        "",
        "import os",
        "from pathlib import Path",
        "",
        "class Foo:",
        "    pass",
        "",
        "def bar():",
        "    pass",
        "",
        "def baz():",
        "    pass",
        "",
        "def qux():",
        "    pass",
        "",
        "x = 1",
        "y = 2",
        "",
        '"""Another module."""',
        "",
        "import sys",
        "",
        "class Bar:",
        "    pass",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) >= 2
    for ch in chunks:
        assert ch.filename_hint is None


# -- split_into_chunks for Python ---------------------------------------------

def test_split_python_module_docstrings():
    """Unindented (column 0) docstrings split into separate chunks."""
    ctx = "\n".join([
        '"""First module: does X."""',
        "",
        "import os",
        "",
        "def foo():",
        "    return 1",
        "",
        "def bar():",
        "    return 2",
        "",
        "def baz():",
        "    return 3",
        "",
        "def qux():",
        "    return 4",
        "",
        "def quux():",
        "    return 5",
        "",
        "",
        '"""Second module: does Y."""',
        "",
        "import sys",
        "",
        "def hello():",
        "    return 6",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2
    assert '"""First module' in chunks[0].lines[0]
    assert '"""Second module' in chunks[1].lines[0]


def test_split_python_multiline_module_docstring():
    """Multiline module docstring at column 0 triggers boundary."""
    ctx = "\n".join([
        '"""',
        "First module.",
        "Does stuff.",
        '"""',
        "",
        "import os",
        "",
        "class Foo:",
        "    pass",
        "",
        "def bar():",
        "    pass",
        "",
        "def baz():",
        "    pass",
        "",
        "def qux():",
        "    pass",
        "",
        "x = 1",
        "y = 2",
        "",
        "",
        '"""',
        "Second module.",
        "Does other stuff.",
        '"""',
        "",
        "import sys",
        "",
        "class Bar:",
        "    pass",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2


def test_split_python_indented_docstring_no_split():
    """Indented docstrings (function/class level) should NOT trigger a split."""
    ctx = "\n".join([
        '"""Module docstring."""',
        "",
        "import os",
        "",
        "class Foo:",
        '    """Class docstring."""',
        "",
        "    def method(self):",
        '        """Method docstring."""',
        "        pass",
        "",
        "    def other(self):",
        '        """Another docstring."""',
        "        pass",
        "",
        "def standalone():",
        '    """Function docstring."""',
        "    pass",
        "",
        "def another():",
        "    pass",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 1


def test_split_python_from_future_boundary():
    """from __future__ import triggers a file boundary."""
    ctx = "\n".join([
        '"""First file."""',
        "",
        "import os",
        "",
        "def foo():",
        "    return 1",
        "",
        "def bar():",
        "    return 2",
        "",
        "def baz():",
        "    return 3",
        "",
        "def qux():",
        "    return 4",
        "",
        "def quux():",
        "    return 5",
        "",
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "",
        "def hello():",
        "    return 6",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2
    assert "from __future__" in chunks[1].lines[0]


def test_split_python_shebang_boundary():
    """Shebang line triggers a file boundary."""
    ctx = "\n".join([
        '"""First file."""',
        "",
        "import os",
        "",
        "def foo():",
        "    return 1",
        "",
        "def bar():",
        "    return 2",
        "",
        "def baz():",
        "    return 3",
        "",
        "def qux():",
        "    return 4",
        "",
        "def quux():",
        "    return 5",
        "",
        "",
        "#!/usr/bin/env python",
        '"""Script file."""',
        "",
        "import sys",
        "",
        "def main():",
        "    pass",
    ])
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2
    assert chunks[1].lines[0].startswith("#!")


# -- language-aware splitting integration ------------------------------------

def test_cpp_boundaries_only_for_cpp():
    """Comment + #include boundaries should NOT trigger for Python contexts."""
    # This context has Python imports so detect_language returns "py".
    # The // comment + #include pattern should be ignored.
    ctx = "\n".join([
        "import os",
        "from pathlib import Path",
        "",
        "class Foo:",
        "    pass",
        "",
        "def bar():",
        "    pass",
        "",
        "def baz():",
        "    pass",
        "",
        "def qux():",
        "    pass",
        "",
        "x = 1",
        "",
        "",
        "// some C comment",
        "// another line",
        "",
        "#include <fake.h>",
        "",
        "more_python = True",
    ])
    assert detect_language(ctx) == "py"
    chunks = split_into_chunks(ctx)
    # Should NOT split at the // + #include since this is Python
    assert all(ch.filename_hint is None for ch in chunks)


def _pad_cpp_chunk(lines, n=25):
    """Pad a list of C++ lines with filler to exceed min_chunk_lines."""
    filler = [f"    int var_{i} = {i};" for i in range(n - len(lines))]
    return lines + filler


def test_mixed_context_detected_as_cpp_splits_correctly():
    """A C++ context with cmake header still splits at comment+#include."""
    cmake = _pad_cpp_chunk([
        "cmake_minimum_required(VERSION 3.13)",
        "project(test)",
        "",
        "set(CMAKE_CXX_STANDARD 17)",
        "add_executable(test main.cpp)",
        "target_link_libraries(test lib)",
        "install(TARGETS test)",
    ])
    file1 = _pad_cpp_chunk([
        "//***************",
        "// Copyright 2024",
        "//***************",
        "",
        "#include <types.h>",
        "#include <system.h>",
        "",
        "class Binning {",
        "    int x;",
        "};",
    ])
    file2 = _pad_cpp_chunk([
        "//***************",
        "// Another file",
        "//***************",
        "",
        "#include <comm.h>",
        "#include <mpi.h>",
        "",
        "class Comm {",
        "    int rank;",
        "};",
    ])
    ctx = "\n".join(cmake + ["", ""] + file1 + ["", ""] + file2)
    assert detect_language(ctx) == "cpp"
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 3
    assert chunks[0].filename_hint is None  # cmake
    assert chunks[1].filename_hint == "types.h"
    assert chunks[2].filename_hint == "comm.h"


def test_language_detection_feeds_into_splitting():
    """detect_language result affects which boundaries are found."""
    def cpp_file(header):
        return _pad_cpp_chunk([
            "//***************",
            "// File header",
            "//***************",
            "",
            f"#include <{header}>",
            "#include <bar.h>",
            "",
            "struct Foo {",
            "    int x;",
            "};",
        ])

    ctx = "\n".join(cpp_file("foo.h") + ["", ""] + cpp_file("baz.h"))
    assert detect_language(ctx) == "cpp"
    chunks = split_into_chunks(ctx)
    assert len(chunks) == 2
    assert chunks[0].filename_hint == "foo.h"
    assert chunks[1].filename_hint == "baz.h"
