from __future__ import annotations


def step_readme(step_title: str, input_summary: str, output_summary: str, technique_summary: str) -> str:
    return (
        f"# {step_title}\n\n"
        f"## 本步做了什么\n\n{step_title} 这一步负责整理并产出当前阶段需要的标准化结果，"
        "并把配置、日志、表格、图像和中间产物归档到统一目录中。\n\n"
        f"## 输入来自哪里\n\n{input_summary}\n\n"
        f"## 输出写到了哪里\n\n{output_summary}\n\n"
        f"## 关键技术与参数\n\n{technique_summary}\n"
    )
