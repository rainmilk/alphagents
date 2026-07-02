#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM client wrapper for OpenAI API.

This module provides a wrapper around the OpenAI API with retry logic,
error handling, and structured response parsing.
"""

import json
import re
import time
from typing import Dict, Optional, Tuple, List, Any
from openai import OpenAI

from .prompts import (
    get_initial_portrait_prompt,
    get_refinement_portrait_prompt,
    get_formula_from_portrait_prompt,
    DIMENSION_GUIDANCE
)
from .example_selector import FewShotExampleSelector
from .performance_aware import (
    get_performance_context,
    adjust_refinement_temperature,
    get_refinement_constraints
)


class LLMClient:
    """
    Wrapper for OpenAI API with domain-specific methods for alpha generation.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 max_retries: int = 3, retry_delay: float = 1.0,
                 base_url: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            base_url: Optional base URL for OpenAI API (e.g. proxy/custom endpoint)
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # 初始化Few-shot示例选择器
        self.example_selector = FewShotExampleSelector(k=3)
    
    def _call_with_retry(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call OpenAI API with exponential backoff retry.
        
        Args:
            prompt: Prompt to send
            temperature: Sampling temperature
            
        Returns:
            Response text
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"API调用失败 (尝试 {attempt + 1}): {e}")
                    print(f"{wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def generate_alpha_portrait(self, context: str = "initial", 
                              dimension: Optional[str] = None,
                              parent_formula: Optional[str] = None,
                              avoid_patterns: Optional[List[str]] = None,
                              examples: Optional[List[Dict]] = None,
                              node_context: Optional[Dict] = None) -> str:
        """
        Generate alpha portrait (high-level description with pseudo code).
        
        Args:
            context: "initial" or "refinement"
            dimension: Refinement dimension (if context is "refinement")
            parent_formula: Parent formula to refine
            avoid_patterns: Patterns to avoid
            
        Returns:
            Alpha portrait text
        """
        if context == "initial":
            prompt = get_initial_portrait_prompt(avoid_patterns)
            temp = 1.0
        else:
            prompt = get_refinement_portrait_prompt(dimension, parent_formula, avoid_patterns, examples, node_context)
            temp = 0.9
        
        response = self._call_with_retry(prompt, temperature=temp)
        
        # Parse JSON response
        try:
            # 尝试提取JSON部分（处理额外的文本）
            json_text = response
            
            # 如果响应被包裹在代码块中
            if '```json' in response:
                match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    json_text = match.group(1)
            elif '```' in response:
                match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    json_text = match.group(1)
            
            # 尝试找到JSON对象
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            
            portrait_data = json.loads(json_text)
            portrait = f"""### Alpha Factor Portrait

**Alpha Name:** {portrait_data.get('name', 'unknown')}

**Description:** {portrait_data.get('description', '')}

**Formula Logic:**
```
{chr(10).join(portrait_data.get('pseudo_code', []))}
```"""
            return portrait
        except json.JSONDecodeError as e:
            # If parsing fails, try to extract key information manually
            print(f"警告: JSON解析失败 ({e})，尝试手动提取信息")
            
            # 尝试手动提取信息
            name_match = re.search(r'"name":\s*"([^"]+)"', response)
            desc_match = re.search(r'"description":\s*"([^"]+)"', response)
            
            if name_match and desc_match:
                portrait = f"""### Alpha Factor Portrait

**Alpha Name:** {name_match.group(1)}

**Description:** {desc_match.group(1)}

**Formula Logic:**
```
# 无法解析伪代码
```"""
                return portrait
            else:
                print("警告: 无法解析响应，返回默认格式")
                return response
    
    def validate_brackets(self, formula: str) -> bool:
        """验证括号是否匹配"""
        stack = []
        for char in formula:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0
    
    def generate_formula_from_portrait(self, portrait: str, 
                                     avoid_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert alpha portrait to symbolic formula with parameter candidates.
        
        根据论文要求，生成符号化公式和3组候选参数。
        
        Args:
            portrait: Alpha portrait text
            avoid_patterns: Patterns to avoid
            
        Returns:
            Dictionary containing symbolic formula, parameters, and candidates
        """
        import json
        import re
        
        # Extract pseudo code from portrait
        pseudo_code = ""
        if "Formula Logic:" in portrait:
            start_idx = portrait.find("```") + 3
            end_idx = portrait.rfind("```")
            if start_idx > 2 and end_idx > start_idx:
                pseudo_code = portrait[start_idx:end_idx].strip()
        
        prompt = get_formula_from_portrait_prompt(portrait, pseudo_code, avoid_patterns)
        response = self._call_with_retry(prompt, temperature=0.7)
        
        # Parse JSON response
        try:
            # Extract JSON from response if wrapped in code blocks
            if '```json' in response:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
            elif '```' in response:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
            
            result = json.loads(response.strip())
            
            # 验证结果结构
            if not all(key in result for key in ['formula', 'parameters', 'candidates']):
                raise ValueError("响应缺少必要字段")
            
            # 验证括号匹配
            if not self.validate_brackets(result['formula']):
                print(f"警告: 公式括号不匹配: {result['formula']}")
                # 尝试修复简单的括号问题
                formula = result['formula']
                # 计算左右括号数量
                left_count = formula.count('(')
                right_count = formula.count(')')
                if left_count > right_count:
                    # 缺少右括号，在末尾添加
                    formula += ')' * (left_count - right_count)
                    print(f"自动修复: 在末尾添加了{left_count - right_count}个右括号")
                    result['formula'] = formula
                elif right_count > left_count:
                    # 右括号过多，尝试在开头添加左括号
                    formula = '(' * (right_count - left_count) + formula
                    print(f"自动修复: 在开头添加了{right_count - left_count}个左括号")
                    result['formula'] = formula
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response[:200]}...")
            
            # 尝试手动提取公式
            formula_match = re.search(r'"formula":\s*"([^"]+)"', response)
            if formula_match:
                formula = formula_match.group(1)
                print(f"手动提取的公式: {formula}")
                
                # 返回包含提取公式的结构
                return {
                    "formula": formula,
                    "parameters": {
                        "w1": {"description": "参数1", "range": [5, 30]},
                        "w2": {"description": "参数2", "range": [10, 60]},
                        "w3": {"description": "参数3", "range": [5, 20]}
                    },
                    "candidates": [
                        {"w1": 20, "w2": 30, "w3": 10},
                        {"w1": 10, "w2": 20, "w3": 5},
                        {"w1": 15, "w2": 45, "w3": 15}
                    ]
                }
            
            # 如果无法提取，返回默认结构
            print("使用默认公式结构")
            return {
                "formula": "Rank(($close - Ref($close, w1)) / Std($close, w2), w3)",
                "parameters": {
                    "w1": {"description": "价格回看窗口", "range": [5, 30]},
                    "w2": {"description": "波动率窗口", "range": [10, 60]},
                    "w3": {"description": "排名窗口", "range": [5, 20]}
                },
                "candidates": [
                    {"w1": 20, "w2": 30, "w3": 10},
                    {"w1": 10, "w2": 20, "w3": 5},
                    {"w1": 15, "w2": 45, "w3": 15}
                ]
            }
        except Exception as e:
            print(f"解析失败（其他错误）: {e}")
            print(f"原始响应: {response[:200]}...")
            
            # 返回默认结构
            return {
                "formula": "Rank(($close - Ref($close, w1)) / Std($close, w2), w3)",
                "parameters": {
                    "w1": {"description": "价格回看窗口", "range": [5, 30]},
                    "w2": {"description": "波动率窗口", "range": [10, 60]},
                    "w3": {"description": "排名窗口", "range": [5, 20]}
                },
                "candidates": [
                    {"w1": 20, "w2": 30, "w3": 10},
                    {"w1": 10, "w2": 20, "w3": 5},
                    {"w1": 15, "w2": 25, "w3": 8}
                ]
            }
    
    def substitute_parameters(self, symbolic_formula: str, params: Dict[str, int]) -> str:
        """
        将符号公式中的参数替换为具体数值。
        
        Args:
            symbolic_formula: 符号化的公式
            params: 参数字典
            
        Returns:
            具体化的公式
        """
        formula = symbolic_formula
        # 按参数名长度降序排序，避免w1被w10覆盖的问题
        sorted_params = sorted(params.items(), key=lambda x: len(x[0]), reverse=True)
        for param_name, param_value in sorted_params:
            formula = re.sub(r'\b' + param_name + r'\b', str(param_value), formula)
        return formula
    
    def generate_initial(self, avoid_patterns: Optional[List[str]] = None,
                        max_attempts: int = 3,
                        evaluator: Optional[Any] = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate initial alpha formula with portrait and parameter optimization.
        
        修改版：返回符号公式而非具体公式，保持公式的灵活性。
        
        Args:
            avoid_patterns: Patterns to avoid
            max_attempts: Maximum generation attempts
            evaluator: Formula evaluator function (optional)
            
        Returns:
            Tuple of (symbolic_formula, portrait, formula_info)
        """
        for attempt in range(max_attempts):
            try:
                print("\n【第1步】LLM生成Alpha画像...")
                portrait = self.generate_alpha_portrait("initial", 
                                                      avoid_patterns=avoid_patterns)
                print(f"生成的Alpha画像:\n{'-' * 40}")
                print(portrait)
                print('-' * 40)
                
                # 生成符号公式和候选参数
                print("\n【第2步】将画像转换为符号公式...")
                formula_info = self.generate_formula_from_portrait(portrait, avoid_patterns)
                print(f"符号公式: {formula_info['formula']}")
                print(f"\n参数说明:")
                for param, info in formula_info['parameters'].items():
                    print(f"  - {param}: {info['description']} (范围: {info['range']})")
                print(f"\n候选参数组:")
                for i, params in enumerate(formula_info['candidates'], 1):
                    print(f"  组{i}: {params}")
                
                # 选择最优参数组，但返回符号公式
                best_score = -1
                best_params = None
                best_scores = None
                
                if evaluator:
                    print("\n【第3步】评估候选参数组...")
                    # 评估每组候选参数
                    for i, params in enumerate(formula_info['candidates']):
                        concrete_formula = self.substitute_parameters(
                            formula_info['formula'], params
                        )
                        print(f"\n评估参数组{i+1}:")
                        print(f"  参数: {params}")
                        print(f"  符号公式: {formula_info['formula']}")
                        print(f"  具体公式（替换前）: {concrete_formula}")
                        
                        # 使用sanitize_formula和fix_missing_params清理公式
                        from ..formula import sanitize_formula, fix_missing_params
                        concrete_formula = sanitize_formula(concrete_formula)
                        concrete_formula = fix_missing_params(concrete_formula)
                        print(f"  具体公式（清理后）: {concrete_formula}")
                        
                        try:
                            result = evaluator(concrete_formula, [], None)
                            if len(result) == 3:
                                scores, _, _ = result  # 忽略raw_scores
                            else:
                                scores, _ = result
                            if scores:
                                avg_score = sum(scores.values()) / len(scores)
                                print(f"  评分详情:")
                                for dim, score in scores.items():
                                    print(f"    - {dim}: {score:.2f}")
                                print(f"  平均分: {avg_score:.2f}")
                                
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = params
                                    best_scores = scores
                        except Exception as e:
                            print(f"  评估失败: {e}")
                else:
                    # 没有评估器时，使用第一组参数
                    best_params = formula_info['candidates'][0]
                
                if best_params:
                    print(f"\n【结果】最优参数组: {best_params}")
                    print(f"符号公式将用于MCTS搜索: {formula_info['formula']}")
                    
                    # 更新formula_info以记录选择的参数和评分
                    formula_info['selected_params'] = best_params
                    formula_info['best_scores'] = best_scores
                    formula_info['symbolic_formula'] = formula_info['formula']
                    
                    # 返回符号公式而非具体公式
                    return formula_info['formula'], portrait, formula_info
                
            except Exception as e:
                print(f"生成失败 (尝试 {attempt+1}/{max_attempts}): {e}")
        
        # 备用方案
        print("使用默认公式")
        default_formula = "Rank(($close - Ref($close, 20)) / Std($close, 30), 10)"
        default_portrait = "### Default Alpha\n\nSimple momentum factor"
        default_info = {
            "formula": "Rank(($close - Ref($close, w1)) / Std($close, w2), w3)",
            "parameters": {
                "w1": {"description": "回看窗口", "range": [5, 30]},
                "w2": {"description": "波动率窗口", "range": [10, 60]},
                "w3": {"description": "排名窗口", "range": [5, 20]}
            },
            "candidates": [{"w1": 20, "w2": 30, "w3": 10}],
            "selected_params": {"w1": 20, "w2": 30, "w3": 10},
            "concrete_formula": default_formula
        }
        return default_formula, default_portrait, default_info
    
    def refine_formula(self, node: Any, dimension: str, 
                      avoid_patterns: Optional[List[str]] = None,
                      repo_examples: Optional[List[Dict]] = None,
                      node_context: Optional[Dict] = None,
                      evaluator: Optional[Any] = None,
                      max_attempts: int = 3) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        沿指定维度细化公式，使用符号参数机制。
        
        修改版：使用符号公式进行精炼，确保结构性变化。
        
        参数：
            node: 包含当前公式的MCTS节点
            dimension: 要细化的维度
            avoid_patterns: 要避免的模式
            repo_examples: 用于上下文的仓库示例
            node_context: 综合节点上下文（父节点、兄弟节点、子节点）
            evaluator: 公式评估器
            max_attempts: 最大细化尝试次数
            
        返回：
            元组 (symbolic_formula, portrait, refinement_description, formula_info)
        """
        # 获取节点的符号公式（如果存在）
        symbolic_formula = node.formula
        if hasattr(node, 'formula_info') and node.formula_info:
            symbolic_formula = node.formula_info.get('symbolic_formula', node.formula)
        
        for attempt in range(max_attempts):
            try:
                # 使用示例选择器选择Few-shot示例
                selected_examples = []
                if repo_examples and hasattr(node, 'factor') and node.factor is not None:
                    # 从repo_examples中提取repository和repo_factors
                    repository = repo_examples.get('repository', [])
                    repo_factors = repo_examples.get('repo_factors', [])
                    
                    # 选择合适的示例
                    selected_examples = self.example_selector.select_examples(
                        dimension=dimension,
                        current_factor=node.factor,
                        repository=repository,
                        repo_factors=repo_factors
                    )
                    
                    if selected_examples:
                        print(f"\n选择了{len(selected_examples)}个Few-shot示例用于{dimension}维度")
                
                # 获取性能感知的精炼策略
                strategy, performance_guidance = get_performance_context(node)
                print(f"\n  使用{strategy}策略进行{dimension}维度优化")
                
                # 调整温度参数
                temperature = adjust_refinement_temperature(node, dimension)
                
                # 在node_context中添加性能指导
                if node_context is None:
                    node_context = {}
                node_context['performance_guidance'] = performance_guidance
                node_context['refinement_strategy'] = strategy
                
                # 生成细化的画像，使用符号公式而非具体公式
                print(f"\n  LLM正在为{dimension}维度生成优化方案...")
                portrait = self.generate_alpha_portrait(
                    "refinement", 
                    dimension=dimension,
                    parent_formula=symbolic_formula,  # 使用符号公式
                    avoid_patterns=avoid_patterns,
                    examples=selected_examples,
                    node_context=node_context
                )
                
                # 生成符号公式和候选参数
                print("  将优化方案转换为新公式...")
                formula_info = self.generate_formula_from_portrait(portrait, avoid_patterns)
                print(f"  新符号公式: {formula_info['formula']}")
                
                # 检查是否有结构性变化
                if formula_info['formula'] != symbolic_formula:
                    # 尝试使用公式对比工具
                    try:
                        from ..utils.formula_diff import highlight_differences
                        old_high, new_high, stats = highlight_differences(symbolic_formula, formula_info['formula'])
                        print(f"\n  🔍 公式变化:")
                        print(f"     原公式: {symbolic_formula}")
                        print(f"     新公式: {formula_info['formula']}")
                        print(f"     结构变化率: {stats['structure_change_ratio']*100:.1f}%")
                        if stats['structure_change_ratio'] > 0.1:
                            print(f"  ✓ 公式结构发生实质性变化")
                        else:
                            print(f"  ⚠ 仅微小调整")
                    except:
                        print(f"  ✓ 公式结构发生变化（不仅是参数调整）")
                else:
                    print(f"  ⚠ 警告：公式结构未变化（仅参数不同）")
                
                # 选择最优参数组，但返回符号公式
                best_score = -1
                best_params = None
                best_scores = None
                
                if evaluator:
                    # 评估每组候选参数
                    for i, params in enumerate(formula_info['candidates']):
                        concrete_formula = self.substitute_parameters(
                            formula_info['formula'], params
                        )
                        
                        try:
                            result = evaluator(concrete_formula, [], node)
                            if len(result) == 3:
                                scores, _, _ = result  # 忽略raw_scores
                            else:
                                scores, _ = result
                            if scores:
                                avg_score = sum(scores.values()) / len(scores)
                                print(f"参数组{i+1}评分: {avg_score:.2f}")
                                
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = params
                                    best_scores = scores
                        except Exception:
                            pass
                else:
                    # 没有评估器时，使用第一组参数
                    best_params = formula_info['candidates'][0]
                
                if best_params:
                    # 更新formula_info
                    formula_info['selected_params'] = best_params
                    formula_info['best_scores'] = best_scores
                    formula_info['symbolic_formula'] = formula_info['formula']
                    
                    # 提取细化描述
                    desc_match = re.search(r'\*\*Description:\*\* (.+?)(?:\n|$)', portrait)
                    if not desc_match:
                        desc_match = re.search(r'Description: (.+?)(?:\n|$)', portrait)
                    refinement_desc = desc_match.group(1) if desc_match else f"针对{dimension}进行了细化"
                    
                    # 返回符号公式而非具体公式
                    return formula_info['formula'], portrait, refinement_desc, formula_info
                
            except Exception as e:
                print(f"优化失败 (尝试 {attempt+1}/{max_attempts}): {e}")
        
        # 备用简单细化
        print("使用简单优化")
        if dimension == "Stability":
            new_formula = f"Mean({node.formula}, 20)"
        elif dimension == "Turnover":
            new_formula = f"Mean({node.formula}, 30)"
        elif dimension == "Diversity":
            new_formula = f"Rank({node.formula}, 10) * Sign(Delta($volume, 5))"
        else:
            new_formula = f"Rank({node.formula}, 5)"
        
        # 创建默认的formula_info
        default_info = {
            "formula": new_formula,
            "parameters": {},
            "candidates": [{}],
            "selected_params": {},
            "concrete_formula": new_formula
        }
        
        return new_formula, f"{dimension}的简单细化", f"应用了简单的{dimension}改进", default_info
    
    def generate_refinement_summary(self, parent_node: Any, child_node: Any,
                                  dimension: str, refinement_desc: str) -> str:
        """
        使用LLM生成综合的细化摘要。
        
        基于算法第27行: L.GenerateRefinementSummary
        
        参数：
            parent_node: 父MCTS节点
            child_node: 细化后的子MCTS节点
            dimension: 被细化的维度
            refinement_desc: 细化的描述
            
        返回：
            综合摘要字符串
        """
        # 计算分数变化
        score_changes = {}
        if parent_node.scores and child_node.scores:
            for dim in parent_node.scores:
                if dim in child_node.scores:
                    change = child_node.scores[dim] - parent_node.scores[dim]
                    score_changes[dim] = f"{change:+.2f}"
        
        prompt = f"""请为以下Alpha公式优化生成简洁的总结。

原始公式：{parent_node.formula}
优化后公式：{child_node.formula}

优化维度：{dimension}
优化描述：{refinement_desc}

分数变化：
{chr(10).join(f'- {k}: {v}' for k, v in score_changes.items())}

请生成一个简洁的总结（1-2句话），说明：
1. 具体做了什么优化
2. 优化带来的主要改进
3. 对整体性能的影响

直接输出总结内容，不要包含其他格式。"""

        try:
            response = self._call_with_retry(prompt, temperature=0.5)
            return response.strip()
        except Exception as e:
            print(f"生成总结失败: {e}")
            # 备用简单摘要
            return f"通过{refinement_desc}优化了{dimension}维度，整体评分从{parent_node.value:.2f}提升到{child_node.value:.2f}。"
    
    def get_completion_sync(self, prompt: str, temperature: float = 0.7, 
                           max_tokens: Optional[int] = None) -> str:
        """
        用于兼容性的同步完成方法。
        
        参数：
            prompt: 要发送的提示
            temperature: 采样温度
            max_tokens: 要生成的最大token数
            
        返回：
            响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()