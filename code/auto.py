import abc # 用于定义抽象基类

class Automaton(abc.ABC):
    """
    自动机的抽象基类，定义所有自动机共有的属性和方法。
    """
    def __init__(self, Q, T, q0, F):
        """
        初始化自动机基本属性。

        Args:
            Q (set): 状态集合。
            T (set): 输入字母表。
            q0: 起始状态。
            F (set): 接受状态集合。
        """
        if not isinstance(Q, set):
            raise TypeError("Q must be a set")
        if not isinstance(T, set):
            raise TypeError("T must be a set")
        if q0 not in Q:
            # 在 DFA 转换中，起始状态可能是一个 frozenset，初始时可能不在 Q 中
            # 放宽这个检查，或者在子类中处理
            pass
            # raise ValueError("Start state must be in the set of Q")
        if not F.issubset(Q):
             # 同样，接受状态在转换过程中可能动态添加
             pass
            # raise ValueError("Accept Q must be a subset of Q")

        self.Q = Q
        self.T = T
        self.q0 = q0
        self.F = F
        # 转移函数由子类具体定义和初始化
        self.transitions = self._initialize_transitions()

    @abc.abstractmethod
    def _initialize_transitions(self):
        #初始化转移函数
        pass

    @abc.abstractmethod
    def add_transition(self, *args):
        #添加转移规则
        pass

    @abc.abstractmethod
    def get_next_Q(self, current_state, symbol):
        """根据当前状态和输入符号获取下一个状态（由子类实现）。"""
        pass

    # 提供一个通用的 __repr__，子类可以覆盖以提供更详细的信息
    def __repr__(self):
        return (f"{type(self).__name__}(\n"
                f"  Q={self.Q},\n"
                f"  T={self.T},\n"
                f"  transitions=... # Specific details in subclass repr\n"
                f"  q0={self.q0},\n"
                f"  F={self.F}\n"
                f")")

class EpsilonNFA(Automaton):
    """
    带有 Epsilon 转移的非确定性有限自动机 (ε-NFA)。
    转移函数格式: {(state, symbol or ''): set_of_next_Q}
    """
    EPSILON = '' # 定义 Epsilon 符号为空字符串，类似#define

    def _initialize_transitions(self):
        return {} # 字典，键是 (状态, 符号或EPSILON)，值是下一状态集合

    def add_transition(self, from_state, symbol, to_Q):
        """
        添加 ε-NFA 转移。

        Args:
            from_state: 起始状态。
            symbol: 输入符号或 EpsilonNFA.EPSILON。
            to_Q (set): 目标状态集合。
        """
        if from_state not in self.Q:
            raise ValueError(f"Invalid from_state: {from_state}")
        if symbol != self.EPSILON and symbol not in self.T:
            raise ValueError(f"Invalid symbol: {symbol}")
        if not isinstance(to_Q, set):
            raise TypeError("to_Q must be a set")
        # 允许添加转换到不在初始状态集的状态，但它们最终应该被加到 Q 集合中
        # if not to_Q.issubset(self.Q):
        #      raise ValueError(f"Invalid to_Q: {to_Q}")

        key = (from_state, symbol)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].update(to_Q)
        # 确保所有涉及的状态都在状态集中
        self.Q.add(from_state)
        self.Q.update(to_Q)


    def get_next_Q(self, current_state, symbol):
        """获取给定状态和符号（包括ε）的下一状态集合。"""
        return self.transitions.get((current_state, symbol), set()) #尝试在函数中查找下一状态，如果没有返回空集set()

    def get_epsilon_closure(self, state_or_Q): #这里的 state_or_Q 可以是单个状态或状态集合
        """计算单个状态或状态集合的 Epsilon 闭包。"""
        closure = set() # 闭包最终的集合状态
        Q_to_process = [] # 通过队列进行处理（待处理列表）

        if isinstance(state_or_Q, set): # 状态集合
            closure.update(state_or_Q)
            Q_to_process.extend(list(state_or_Q)) # 将集合中的所有状态加入待处理列表
        else: # 单个状态
            closure.add(state_or_Q)
            Q_to_process.append(state_or_Q)

        processed = set() # 保存已处理过的状态，用于避免重复处理和无限循环

        while Q_to_process: # 依次处理所有状态
            current = Q_to_process.pop(0) # 使用队列方式处理
            if current in processed:
                continue
            processed.add(current)

            epsilon_next = self.get_next_Q(current, self.EPSILON)
            new_Q = epsilon_next - closure # 只添加尚未在闭包中的新状态
            if new_Q:
                closure.update(new_Q)
                Q_to_process.extend(list(new_Q)) # 将新状态加入待处理列表

        return closure

    def __repr__(self):
        # 提供更详细的 transitions 显示
        trans_str = "{\n"
        # 对转移进行排序以便于查看
        sorted_transitions = sorted(self.transitions.items(), key=lambda item: (item[0][0], item[0][1]))
        for (state, symbol), next_Q in sorted_transitions:
             symbol_repr = f"'{symbol}'" if symbol != self.EPSILON else "ε"
             # 对 next_Q 排序
             sorted_next_Q = sorted(list(next_Q))
             trans_str += f"    ({state}, {symbol_repr}): {set(sorted_next_Q)},\n"
        trans_str += "  }"
        return (f"{type(self).__name__}(\n"
                f"  Q={sorted(list(self.Q))},\n" # 排序状态
                f"  T={sorted(list(self.T))},\n" # 排序字母表
                f"  transitions={trans_str},\n"
                f"  q0={self.q0},\n"
                f"  F={sorted(list(self.F))}\n" # 排序接受状态
                f")")

class NFA(EpsilonNFA): # 继承自 EpsilonNFA
    """
    非确定性有限自动机 (NFA)。
    可以看作是没有 Epsilon 转移的 EpsilonNFA。
    转移函数格式: {(state, symbol): set_of_next_Q}
    """
    def add_transition(self, from_state, symbol, to_Q):
        #不添加 Epsilon 转移，与 EpsilonNFA 区别
        if symbol == self.EPSILON:
            raise ValueError("NFA cannot have Epsilon transitions. Use EpsilonNFA class instead.")
        super().add_transition(from_state, symbol, to_Q)

    # 对于标准 NFA，epsilon 闭包就是其自身
    # 这个函数似乎没有必要，因为 NFA 不使用 Epsilon 转移
    def get_epsilon_closure(self, state_or_Q):
        if isinstance(state_or_Q, set):
            return state_or_Q.copy()
        else:
            return {state_or_Q}

    # __repr__ 可以继承自 EpsilonNFA，因为它能正确处理非 Epsilon 转移

class DFA(Automaton):
    """
    确定性有限自动机 (DFA)。
    转移函数格式: {(state, symbol): single_next_state}
    DFA 的状态通常是 NFA 状态的集合 (表示为 frozenset)。
    """
    def _initialize_transitions(self):
        return {} # 字典，键是 (状态, 符号)，值是单个下一状态

    def add_transition(self, from_state, symbol, to_state):
        """
        添加 DFA 转移。

        Args:
            from_state: 起始状态。
            symbol: 输入符号。
            to_state: 目标状态 (单个状态)。
        """
        # DFA 状态通常是 frozenset，检查类型可能复杂，依赖于具体实现
        if from_state not in self.Q:
            raise ValueError(f"Invalid from_state: {from_state}")
        if symbol not in self.T:
             raise ValueError(f"Invalid symbol: {symbol}")
        # if to_state not in self.Q:
        #     # 允许在构建过程中添加新状态
        #     pass

        key = (from_state, symbol)
        if key in self.transitions and self.transitions[key] != to_state:
            # 检查确定性：同一个输入不应有多个不同的下一状态
            raise ValueError(f"Non-deterministic transition attempted for DFA: {key} -> {to_state} (already maps to {self.transitions[key]})")

        self.transitions[key] = to_state
        # 确保所有涉及的状态都在状态集中
        self.Q.add(from_state)
        self.Q.add(to_state)


    def get_next_Q(self, current_state, symbol):
        """获取 DFA 的下一个状态。返回单个状态或 None (如果未定义)。"""
        # 返回集合以便与 NFA 兼容？或者坚持返回单个状态/None？
        # 为了清晰区分 DFA，返回单个状态或 None 更好。
        # 如果需要集合形式，调用者可以自己包装。
        return self.transitions.get((current_state, symbol))

    # 可以添加检查 DFA 是否完整的方法
    def is_complete(self):
        """检查 DFA 是否是完整的（每个状态对每个字母表符号都有转移）。"""
        for state in self.Q:
            for symbol in self.T:
                if (state, symbol) not in self.transitions:
                    return False
        return True

    def __repr__(self):
        # 自定义状态显示，特别是对于 frozenset 状态
        def state_to_str(s):
             if isinstance(s, frozenset):
                 # 排序以获得一致的输出
                 return str(set(sorted(list(s)))) if s else '{}'
             return str(s)

        # 对状态、接受状态进行排序和格式化
        sorted_Q = sorted(list(self.Q), key=state_to_str)
        Q_str = "{" + ", ".join(map(state_to_str, sorted_Q)) + "}"
        sorted_F = sorted(list(self.F), key=state_to_str)
        F_str = "{" + ", ".join(map(state_to_str, sorted_F)) + "}"
        q0_str = state_to_str(self.q0)

        # 对转移进行排序和格式化
        trans_str = "{\n"
        sorted_transitions = sorted(self.transitions.items(), key=lambda item: (state_to_str(item[0][0]), item[0][1]))
        for (state, symbol), next_state in sorted_transitions:
             trans_str += f"    ({state_to_str(state)}, '{symbol}'): {state_to_str(next_state)},\n"
        trans_str += "  }"

        return (f"{type(self).__name__}(\n"
                f"  Q={Q_str},\n"
                f"  T={sorted(list(self.T))},\n" # 排序字母表
                f"  transitions={trans_str},\n"
                f"  q0={q0_str},\n"
                f"  F={F_str}\n"
                f")")

# --- 示例用法 ---
# EpsilonNFA 示例
# enfa_Q = {0, 1, 2, 3}
# enfa_T = {'a', 'b'}
# enfa_start = 0
# enfa_accept = {3}
# enfa = EpsilonNFA(enfa_Q, enfa_T, enfa_start, enfa_accept)
# enfa.add_transition(0, EpsilonNFA.EPSILON, {1})
# enfa.add_transition(1, 'a', {1, 2})
# enfa.add_transition(2, 'b', {3})
# enfa.add_transition(3, EpsilonNFA.EPSILON, {1}) # ε-transition back
# print("--- EpsilonNFA Example ---")
# print(enfa)
# print(f"Epsilon closure of 0: {enfa.get_epsilon_closure(0)}")
# print(f"Epsilon closure of {{0, 2}}: {enfa.get_epsilon_closure({0, 2})}")
# print("-" * 20)

# # NFA 示例 (不能有 epsilon)
# nfa_Q = {'q0', 'q1', 'q2'}
# nfa_T = {'0', '1'}
# nfa_start = 'q0'
# nfa_accept = {'q2'}
# nfa = NFA(nfa_Q, nfa_T, nfa_start, nfa_accept)
# nfa.add_transition('q0', '0', {'q0'})
# nfa.add_transition('q0', '1', {'q0', 'q1'})
# nfa.add_transition('q1', '0', {'q2'})
# nfa.add_transition('q1', '1', {'q2'})
# # nfa.add_transition('q2', nfa.EPSILON, {'q0'}) # 这会引发 ValueError
# print("--- NFA Example ---")
# print(nfa)
# print("-" * 20)

# # DFA 示例 (状态可以是 frozenset)
# dfa_Q = {frozenset({0, 1}), frozenset({2}), frozenset()} # 包含空集状态示例
# dfa_T = {'x', 'y'}
# dfa_start = frozenset({0, 1})
# dfa_accept = {frozenset({2})}
# dfa = DFA(dfa_Q, dfa_T, dfa_start, dfa_accept)
# dfa.add_transition(frozenset({0, 1}), 'x', frozenset({2}))
# dfa.add_transition(frozenset({0, 1}), 'y', frozenset()) # 转移到空集/死状态
# dfa.add_transition(frozenset({2}), 'x', frozenset({2}))
# dfa.add_transition(frozenset({2}), 'y', frozenset({2}))
# dfa.add_transition(frozenset(), 'x', frozenset()) # 死状态的转移
# dfa.add_transition(frozenset(), 'y', frozenset())
# print("--- DFA Example ---")
# print(dfa)
# print(f"Is DFA complete? {dfa.is_complete()}")
# print("-" * 20)