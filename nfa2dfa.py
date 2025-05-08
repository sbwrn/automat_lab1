import abc  # 用于定义抽象基类
import pickle  # 用于保存和加载自动机

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
    
    def accepts(self, input_string):
        """检查自动机是否接受输入字符串（由子类实现）。"""
        pass
    
    def save(self, filename):
        """
        将自动机保存到文件。
        
        Args:
            filename (str): 保存的文件名。
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """
        从文件加载自动机。
        
        Args:
            filename (str): 加载的文件名。
            
        Returns:
            Automaton: 加载的自动机对象。
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    '''
    def visualize(self, filename=None):
        """
        可视化自动机（需要安装graphviz）。
        
        Args:
            filename (str, optional): 输出文件名（不含扩展名）。如果未提供，则仅显示图形。
        """
        try:
            import graphviz
        except ImportError:
            print("Error: graphviz package not installed. Install with 'pip install graphviz'.")
            return
        
        # 创建有向图
        dot = graphviz.Digraph(comment=f'{type(self).__name__} Visualization')
        
        # 添加节点
        for state in self.Q:
            # 确定节点形状：接受状态为双圆，否则为单圆
            shape = 'doublecircle' if state in self.F else 'circle'
            # 确定节点颜色：初始状态为绿色，否则为黑色
            color = 'green' if state == self.q0 else 'black'
            
            # 将状态转换为字符串表示
            if isinstance(state, frozenset):
                label = '{' + ', '.join(map(str, state)) + '}' if state else '∅'
            else:
                label = str(state)
            
            dot.node(str(id(state)), label, shape=shape, color=color)
        
        # 添加转移边
        self._add_transitions_to_graph(dot)
        
        # 渲染图形
        if filename:
            dot.render(filename, view=True)
        else:
            dot.view()
        
        return dot
    
    @abc.abstractmethod
    def _add_transitions_to_graph(self, dot):
        """为可视化添加转移边（由子类实现）。"""
        pass
        '''
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
        self.Q.add(from_state)
        self.Q.update(to_Q)
        
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
    
    def accepts(self, input_string):
        """
        检查ε-NFA是否接受输入字符串。
        
        Args:
            input_string (str): 输入字符串。
            
        Returns:
            bool: 如果自动机接受该字符串则返回True，否则返回False。
        """
        # 起始状态的ε闭包
        current_states = self.get_epsilon_closure(self.q0)
        
        # 遍历输入字符串中的每个符号
        for symbol in input_string:
            if symbol not in self.T:
                return False  # 字符不在字母表中
                
            # 计算下一个状态集合
            next_states = set()
            for state in current_states:
                # 获取当前状态通过当前符号能达到的所有状态
                symbol_states = self.get_next_Q(state, symbol)
                next_states.update(symbol_states)
            
            # 计算这些状态的ε闭包
            current_states = set()
            for state in next_states:
                current_states.update(self.get_epsilon_closure(state))
                
            if not current_states:
                return False  # 无法继续转移
        
        # 检查是否至少有一个接受状态
        return any(state in self.F for state in current_states)
    
    def to_nfa(self):
        """将ε-NFA转换为等价的NFA。"""
        # 创建新的NFA
        nfa = NFA(set(), self.T.copy(), self.q0, set())
        
        # 对于每个状态和每个字母表符号，计算包含ε转移的转移
        for state in self.Q:
            # 计算状态的ε闭包
            epsilon_closure = self.get_epsilon_closure(state)
            
            # 对于字母表中的每个符号
            for symbol in self.T:
                next_states = set()
                
                # 对于ε闭包中的每个状态，找到它们通过symbol可以到达的状态
                for epsilon_state in epsilon_closure:
                    direct_states = self.get_next_Q(epsilon_state, symbol)
                    
                    # 对于每个直接到达的状态，加入它的ε闭包
                    for direct_state in direct_states:
                        next_states.update(self.get_epsilon_closure(direct_state))
                
                # 如果有可到达的状态，添加到NFA中
                if next_states:
                    nfa.add_transition(state, symbol, next_states)
            
            # 如果状态的ε闭包包含任何接受状态，则该状态在NFA中也是接受状态
            if any(s in self.F for s in epsilon_closure):
                nfa.F.add(state)
        
        # 确保所有状态都被添加到NFA中
        nfa.Q.update(self.Q)
        
        return nfa
    '''
    def _add_transitions_to_graph(self, dot):
        """为可视化添加ε-NFA的转移边。"""
        # 收集从相同源到相同目标的多个转移，以便合并显示
        transitions_map = {}
        
        for (state, symbol), next_states in self.transitions.items():
            for next_state in next_states:
                key = (state, next_state)
                if key not in transitions_map:
                    transitions_map[key] = []
                
                # 将ε显示为希腊字母
                label = 'ε' if symbol == self.EPSILON else symbol
                transitions_map[key].append(label)
        
        # 添加边并合并标签
        for (state, next_state), symbols in transitions_map.items():
            # 合并多个符号为一个标签
            label = ', '.join(symbols)
            dot.edge(str(id(state)), str(id(next_state)), label=label)
    '''
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
    
    def accepts(self, input_string):
        """检查NFA是否接受输入字符串。"""
        # NFA的接受功能与EpsilonNFA相同，只是没有ε转移
        return super().accepts(input_string)
    
    def to_dfa(self):
        """将NFA转换为等价的DFA。"""
        # 创建DFA的初始状态集
        dfa_initial = frozenset({self.q0})
        dfa_states = {dfa_initial}  # 存储DFA状态（每个状态是NFA状态的一个子集）
        dfa_accept = set()  # DFA的接受状态
        unprocessed = [dfa_initial]  # 待处理的DFA状态队列
        
        # 创建一个新的DFA
        dfa = DFA(set(), self.T.copy(), dfa_initial, set())
        
        # 处理所有DFA状态
        while unprocessed:
            current_dfa_state = unprocessed.pop(0)
            
            # 对于每个输入符号
            for symbol in self.T:
                next_nfa_states = set()
                
                # 计算通过当前符号可以到达的所有NFA状态
                for nfa_state in current_dfa_state:
                    next_nfa_states.update(self.get_next_Q(nfa_state, symbol))
                
                # 如果有下一个状态，创建新的DFA转移
                if next_nfa_states:
                    # 将NFA状态集转换为DFA状态（frozenset）
                    next_dfa_state = frozenset(next_nfa_states)
                    
                    # 添加转移
                    dfa.add_transition(current_dfa_state, symbol, next_dfa_state)
                    
                    # 如果这是一个新的DFA状态，添加到待处理列表
                    if next_dfa_state not in dfa_states:
                        dfa_states.add(next_dfa_state)
                        unprocessed.append(next_dfa_state)
                else:
                    # 没有下一状态，转移到空集状态（死状态）
                    empty_state = frozenset()
                    dfa.add_transition(current_dfa_state, symbol, empty_state)
                    
                    # 如果死状态是新状态，添加到状态集
                    if empty_state not in dfa_states:
                        dfa_states.add(empty_state)
                        # 处理死状态的转移
                        for sym in self.T:
                            dfa.add_transition(empty_state, sym, empty_state)
        
        # 确定DFA的接受状态
        for dfa_state in dfa_states:
            # 如果DFA状态包含任何NFA接受状态，则它是接受状态
            if any(nfa_state in self.F for nfa_state in dfa_state):
                dfa.F.add(dfa_state)
        
        # 更新DFA的状态集
        dfa.Q = dfa_states
        
        return dfa

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
    
    def accepts(self, input_string):
        """
        检查DFA是否接受输入字符串。
        
        Args:
            input_string (str): 输入字符串。
            
        Returns:
            bool: 如果自动机接受该字符串则返回True，否则返回False。
        """
        current_state = self.q0
        
        # 遍历输入字符串中的每个符号
        for symbol in input_string:
            if symbol not in self.T:
                return False  # 字符不在字母表中
                
            # 获取下一个状态
            current_state = self.get_next_Q(current_state, symbol)
            
            # 如果没有下一个状态，拒绝
            if current_state is None:
                return False
        
        # 检查最终状态是否是接受状态
        return current_state in self.F
    
    # def minimize(self):

# 主程序
def main():
    """
    自动机转换程序的主函数，提供交互式菜单来创建和转换自动机。
    """
    print("=== 有限自动机转换程序 ===")
    
    while True:
        print("\n选择操作:")
        print("1. 创建 EpsilonNFA (带ε转移的非确定性有限自动机)")
        print("2. 创建 NFA (非确定性有限自动机)")
        print("3. 创建 DFA (确定性有限自动机)")
        print("4. 从文件加载自动机")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == '1':
            automaton = create_enfa()
            if automaton:
                process_automaton(automaton)
        elif choice == '2':
            automaton = create_nfa()
            if automaton:
                process_automaton(automaton)
        elif choice == '3':
            automaton = create_dfa()
            if automaton:
                process_automaton(automaton)
        elif choice == '4':
            automaton = load_automaton()
            if automaton:
                process_automaton(automaton)
        elif choice == '5':
            print("程序已退出。")
            break
        else:
            print("无效选项，请重新输入。")

def create_enfa():
    """创建ε-NFA并返回。"""
    print("\n=== 创建 EpsilonNFA ===")
    
    try:
        # 获取状态集
        states_input = input("输入状态集 (用逗号分隔，如: q0,q1,q2): ").strip()
        states = set(states_input.split(','))
        
        # 获取字母表
        alphabet_input = input("输入字母表 (用逗号分隔，如: a,b,c): ").strip()
        alphabet = set(alphabet_input.split(','))
        
        # 获取初始状态
        start_state = input("输入初始状态: ").strip()
        if start_state not in states:
            print(f"警告: 初始状态 {start_state} 不在状态集中。将自动添加。")
            states.add(start_state)
        
        # 获取接受状态
        accept_input = input("输入接受状态集 (用逗号分隔): ").strip()
        accept_states = set(accept_input.split(','))
        for state in accept_states:
            if state not in states:
                print(f"警告: 接受状态 {state} 不在状态集中。将自动添加。")
                states.add(state)
        
        # 创建 EpsilonNFA
        enfa = EpsilonNFA(states, alphabet, start_state, accept_states)
        
        # 添加转移
        print("\n添加转移 (输入空行结束):")
        print("格式: 当前状态,输入符号,目标状态1,目标状态2,... (对于ε转移，输入符号留空)")
        
        while True:
            transition = input("转移: ").strip()
            if not transition:
                break
            
            parts = transition.split(',')
            if len(parts) < 3:
                print("错误: 转移格式无效。请重新输入。")
                continue
            
            from_state = parts[0]
            symbol = parts[1] if parts[1] else EpsilonNFA.EPSILON
            to_states = set(parts[2:])
            
            try:
                enfa.add_transition(from_state, symbol, to_states)
                print(f"已添加转移: ({from_state}, {symbol if symbol else 'ε'}) -> {to_states}")
            except Exception as e:
                print(f"错误: {e}")
        
        print("\nEpsilonNFA 创建成功:")
        print(enfa)
        return enfa
    
    except Exception as e:
        print(f"创建 EpsilonNFA 失败: {e}")
        return None

def create_nfa():
    """创建NFA并返回。"""
    print("\n=== 创建 NFA ===")
    
    try:
        # 获取状态集
        states_input = input("输入状态集 (用逗号分隔，如: q0,q1,q2): ").strip()
        states = set(states_input.split(','))
        
        # 获取字母表
        alphabet_input = input("输入字母表 (用逗号分隔，如: a,b,c): ").strip()
        alphabet = set(alphabet_input.split(','))
        
        # 获取初始状态
        start_state = input("输入初始状态: ").strip()
        if start_state not in states:
            print(f"警告: 初始状态 {start_state} 不在状态集中。将自动添加。")
            states.add(start_state)
        
        # 获取接受状态
        accept_input = input("输入接受状态集 (用逗号分隔): ").strip()
        accept_states = set(accept_input.split(','))
        for state in accept_states:
            if state not in states:
                print(f"警告: 接受状态 {state} 不在状态集中。将自动添加。")
                states.add(state)
        
        # 创建 NFA
        nfa = NFA(states, alphabet, start_state, accept_states)
        
        # 添加转移
        print("\n添加转移 (输入空行结束):")
        print("格式: 当前状态,输入符号,目标状态1,目标状态2,...")
        
        while True:
            transition = input("转移: ").strip()
            if not transition:
                break
            
            parts = transition.split(',')
            if len(parts) < 3:
                print("错误: 转移格式无效。请重新输入。")
                continue
            
            from_state = parts[0]
            symbol = parts[1]
            to_states = set(parts[2:])
            
            try:
                nfa.add_transition(from_state, symbol, to_states)
                print(f"已添加转移: ({from_state}, {symbol}) -> {to_states}")
            except Exception as e:
                print(f"错误: {e}")
        
        print("\nNFA 创建成功:")
        print(nfa)
        return nfa
    
    except Exception as e:
        print(f"创建 NFA 失败: {e}")
        return None

def create_dfa():
    """创建DFA并返回。"""
    print("\n=== 创建 DFA ===")
    
    try:
        # 获取状态集
        states_input = input("输入状态集 (用逗号分隔，如: q0,q1,q2): ").strip()
        states = set(states_input.split(','))
        
        # 获取字母表
        alphabet_input = input("输入字母表 (用逗号分隔，如: a,b,c): ").strip()
        alphabet = set(alphabet_input.split(','))
        
        # 获取初始状态
        start_state = input("输入初始状态: ").strip()
        if start_state not in states:
            print(f"警告: 初始状态 {start_state} 不在状态集中。将自动添加。")
            states.add(start_state)
        
        # 获取接受状态
        accept_input = input("输入接受状态集 (用逗号分隔): ").strip()
        accept_states = set(accept_input.split(','))
        for state in accept_states:
            if state not in states:
                print(f"警告: 接受状态 {state} 不在状态集中。将自动添加。")
                states.add(state)
        
        # 创建 DFA
        dfa = DFA(states, alphabet, start_state, accept_states)
        
        # 添加转移
        print("\n添加转移 (输入空行结束):")
        print("格式: 当前状态,输入符号,目标状态")
        
        while True:
            transition = input("转移: ").strip()
            if not transition:
                break
            
            parts = transition.split(',')
            if len(parts) != 3:
                print("错误: 转移格式无效。请重新输入。")
                continue
            
            from_state = parts[0]
            symbol = parts[1]
            to_state = parts[2]
            
            try:
                dfa.add_transition(from_state, symbol, to_state)
                print(f"已添加转移: ({from_state}, {symbol}) -> {to_state}")
            except Exception as e:
                print(f"错误: {e}")
        
        print("\nDFA 创建成功:")
        print(dfa)
        return dfa
    
    except Exception as e:
        print(f"创建 DFA 失败: {e}")
        return None

def load_automaton():
    """从文件加载自动机。"""
    try:
        filename = input("输入保存的自动机文件名: ").strip()
        automaton = Automaton.load(filename)
        print(f"成功从文件 {filename} 加载自动机。")
        print(automaton)
        return automaton
    except Exception as e:
        print(f"加载自动机失败: {e}")
        return None

def process_automaton(automaton):
    """对自动机进行操作。"""
    while True:
        print("\n选择操作:")
        print("1. 测试接受字符串")
        print("2. 转换为DFA")
        if isinstance(automaton, DFA):
            print("3. 最小化DFA") 
        print("4. 查看自动机")
        print("5. 保存自动机")
        print("6. 可视化自动机")
        print("7. 返回主菜单")
        
        choice = input("\n请输入选项: ").strip()
        
        if choice == '1':
            test_string = input("输入要测试的字符串: ").strip()
            result = automaton.accepts(test_string)
            print(f"自动机{'接受' if result else '拒绝'}字符串 '{test_string}'")
        
        elif choice == '2':
            if isinstance(automaton, DFA):
                print("自动机已经是DFA。")
            else:
                try:
                    if isinstance(automaton, EpsilonNFA) and not isinstance(automaton, NFA):
                        print("将EpsilonNFA转换为NFA，再转换为DFA...")
                        nfa = automaton.to_nfa()
                        dfa = nfa.to_dfa()
                    else:
                        print("将NFA转换为DFA...")
                        dfa = automaton.to_dfa()
                    
                    print("\n转换完成，生成的DFA:")
                    print(dfa)
                    
                    automaton = dfa  # 更新当前工作的自动机
                except Exception as e:
                    print(f"转换失败: {e}")
        
        elif choice == '3' and isinstance(automaton, DFA):
            try:
                print("最小化DFA...")
                min_dfa = automaton.minimize()
                print("\n最小化完成，生成的最小DFA:")
                print(min_dfa)
                
                automaton = min_dfa  # 更新当前工作的自动机
            except Exception as e:
                print(f"最小化失败: {e}")
        
        elif choice == '4':
            print("\n当前自动机:")
            print(automaton)
        
        elif choice == '5':
            try:
                filename = input("输入保存文件名: ").strip()
                automaton.save(filename)
                print(f"自动机已保存到文件 {filename}")
            except Exception as e:
                print(f"保存失败: {e}")
        
        elif choice == '6':
            try:
                filename = input("输入可视化输出文件名 (可选): ").strip()
                automaton.visualize(filename if filename else None)
                print("可视化完成。")
            except Exception as e:
                print(f"可视化失败: {e}")
        
        elif choice == '7':
            break
        
        else:
            print("无效选项，请重新输入。")

def run_example():
    """运行内置示例。"""
    print("\n=== 运行内置示例 ===")
    
    # 创建一个简单的NFA示例
    nfa = NFA({'q0', 'q1', 'q2'}, {'0', '1'}, 'q0', {'q2'})
    nfa.add_transition('q0', '0', {'q0'})
    nfa.add_transition('q0', '1', {'q0', 'q1'})
    nfa.add_transition('q1', '0', {'q2'})
    nfa.add_transition('q1', '1', {'q2'})
    
    print("创建的NFA:")
    print(nfa)
    
    # 测试接受字符串
    test_strings = ['0', '1', '10', '11', '110', '010']
    print("\n测试字符串接受情况:")
    for s in test_strings:
        result = nfa.accepts(s)
        print(f"NFA {'接受' if result else '拒绝'} '{s}'")
    
    # 转换为DFA
    print("\n将NFA转换为DFA...")
    dfa = nfa.to_dfa()
    print("转换后的DFA:")
    print(dfa)
    
    # 测试DFA接受字符串
    print("\n测试转换后的DFA接受情况:")
    for s in test_strings:
        nfa_result = nfa.accepts(s)
        dfa_result = dfa.accepts(s)
        print(f"字符串 '{s}': NFA: {'接受' if nfa_result else '拒绝'}, DFA: {'接受' if dfa_result else '拒绝'}, 等价: {nfa_result == dfa_result}")
    
    '''
    # 最小化DFA
    print("\n最小化DFA...")
    min_dfa = dfa.minimize()
    print("最小化后的DFA:")
    print(min_dfa)
    
    # 测试最小化DFA
    print("\n测试最小化后的DFA接受情况:")
    for s in test_strings:
        dfa_result = dfa.accepts(s)
        min_result = min_dfa.accepts(s)
        print(f"字符串 '{s}': 原DFA: {'接受' if dfa_result else '拒绝'}, 最小DFA: {'接受' if min_result else '拒绝'}, 等价: {dfa_result == min_result}")
    
    return min_dfa

    '''
if __name__ == "__main__":
    print("欢迎使用有限自动机转换程序\n")
    
    print("是否运行内置示例? (y/n)")
    if input().strip().lower() in ['y', 'yes']:
        example_automaton = run_example()
        print("\n是否继续使用该示例自动机? (y/n)")
        if input().strip().lower() in ['y', 'yes']:
            process_automaton(example_automaton)
    
    main()