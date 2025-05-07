+++
author = "Ming"
title = "Tex-Latex-Katex"
date = "2025-04-19"
description = "简单介绍一下Tex、Latex、Katex"
tags = [
    "Katex",
    "Tex",
    "Latex"
]
+++

# TeX、LaTeX 和 KaTeX：排版系统的演进与应用

## TeX：排版系统的基石

TeX 由 Donald Knuth 开发，是一个功能强大的排版系统，特别擅长处理数学公式。它采用"所见即所得"的设计哲学，用户通过编写纯文本指令来控制排版。

TeX 的主要特点：
- 精确控制文档的每个细节
- 强大的数学公式排版能力
- 开源且跨平台
- 需要编译才能生成最终文档

## LaTeX：TeX 的扩展

LaTeX 是建立在 TeX 之上的宏包集合，由 Leslie Lamport 开发。它简化了 TeX 的使用，提供了更高层次的抽象。

LaTeX 的优势：
- 结构化文档编写
- 自动编号和交叉引用
- 丰富的模板和宏包生态
- 学术论文写作的事实标准

```latex
\documentclass{article}
\begin{document}
Hello, \LaTeX!
\end{document}
```

## KaTeX：Web 数学排版

KaTeX 是一个轻量级的 JavaScript 库，用于在网页上快速渲染数学公式。它专注于性能，适合现代 Web 应用。

KaTeX 的特点：
- 即时渲染，无需页面刷新
- 支持大部分 LaTeX 数学语法
- 体积小，加载速度快
- 与 Markdown 兼容

```markdown
当 $a \ne 0$ 时，二次方程 $ax^2 + bx + c = 0$ 的解为：
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}$$
```

## 比较与选择指南

| 特性        | TeX       | LaTeX     | KaTeX     |
|------------|----------|----------|----------|
| 复杂度      | 高        | 中        | 低        |
| 学习曲线    | 陡峭      | 中等      | 平缓      |
| 渲染方式    | 编译      | 编译      | 即时      |
| 适用场景    | 专业排版  | 学术文档  | 网页数学  |

## 总结

TeX 提供了强大的底层排版能力，LaTeX 使其更易用于学术写作，而 KaTeX 则将这些能力带到了 Web 上。根据您的需求：
- 写论文或书籍：选择 LaTeX
- 需要完全控制排版：选择 TeX
- 在网页显示公式：选择 KaTeX
