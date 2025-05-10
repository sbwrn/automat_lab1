import abc  # 用于定义抽象基类
import pickle  # 用于保存和加载自动机
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + r"D:\Graphviz-12.2.1-win64\bin"

class Automaton(abc.ABC):
    def __init__(self, Q, T, q0, F):
        """
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
            pass
        if not F.issubset(Q):
             pass

        self.Q = Q
        self.T = T
        self.q0 = q0
        self.F = F
        self.transitions = self._initialize_transitions()

    @abc.abstractmethod
    def _initialize_transitions(self):
        pass

    @abc.abstractmethod
    def add_transition(self, *args):
        pass

    @abc.abstractmethod
    def get_next_Q(self, current_state, symbol):
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
        pass
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        

#可视化实现 利用graphviz库   
    def visualize(self, filename=None):
        # filename (str, optional): 输出文件名（不含扩展名）。如果未提供，则仅显示图形。
        try:
           import graphviz
        except ImportError:
            print("Error: graphviz package not installed. Install with 'pip install graphviz'.")
            return
    
        dot = graphviz.Digraph(comment=f'{type(self).__name__} Visualization')
    
        state_to_node_id = {}
    
        for state in self.Q:            
            if isinstance(self, DFA) and isinstance(state, frozenset) and not state:
                continue
            shape = 'doublecircle' if state in self.F else 'circle'

            color = 'green' if state == self.q0 else 'black'
            if isinstance(state, frozenset):
                label = '{' + ', '.join(map(str, state)) + '}' if state else '∅'
            else:
                label = str(state)
            node_id = label
            state_to_node_id[state] = node_id
            dot.node(node_id, label, shape=shape, color=color)
        self._add_transitions_to_graph(dot, state_to_node_id)

        if filename:
            dot.render(filename, view=True)
        else:
            dot.view()
    
        return dot

class EpsilonNFA(Automaton):
    """
    带有 Epsilon 转移的非确定性有限自动机 (ε-NFA)。
    转移函数格式: {(state, symbol or ''): set_of_next_Q}
    """
    EPSILON = ''  # 定义 Epsilon 符号为空字符串

    def _initialize_transitions(self):
        return {}

    def add_transition(self, from_state, symbol, to_Q):
        self.Q.add(from_state)
        self.Q.update(to_Q)
        if from_state not in self.Q:
            raise ValueError(f"Invalid from_state: {from_state}")
        if symbol != self.EPSILON and symbol not in self.T:
            raise ValueError(f"Invalid symbol: {symbol}")
        if not isinstance(to_Q, set):
            raise TypeError("to_Q must be a set")
        key = (from_state, symbol)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].update(to_Q)

    def get_next_Q(self, current_state, symbol):
        return self.transitions.get((current_state, symbol), set())

    def get_epsilon_closure(self, state_or_Q):
        closure = set()
        Q_to_process = []
        if isinstance(state_or_Q, set):
            closure.update(state_or_Q)
            Q_to_process.extend(list(state_or_Q))
        else:
            closure.add(state_or_Q)
            Q_to_process.append(state_or_Q)
        processed = set()
        while Q_to_process:
            current = Q_to_process.pop(0)
            if current in processed:
                continue
            processed.add(current)
            epsilon_next = self.get_next_Q(current, self.EPSILON)
            new_Q = epsilon_next - closure
            if new_Q:
                closure.update(new_Q)
                Q_to_process.extend(list(new_Q))
        return closure

    def __repr__(self):
        trans_str = "{\n"
        sorted_transitions = sorted(self.transitions.items(), key=lambda item: (item[0][0], item[0][1]))
        for (state, symbol), next_Q in sorted_transitions:
            symbol_repr = f"'{symbol}'" if symbol != self.EPSILON else "ε"
            sorted_next_Q = sorted(list(next_Q))
            trans_str += f"    ({state}, {symbol_repr}): {set(sorted_next_Q)},\n"
        trans_str += "  }"
        return (f"{type(self).__name__}(\n"
                f"  Q={sorted(list(self.Q))},\n"
                f"  T={sorted(list(self.T))},\n"
                f"  transitions={trans_str},\n"
                f"  q0={self.q0},\n"
                f"  F={sorted(list(self.F))}\n"
                f")")

    def accepts(self, input_string):
        current_states = self.get_epsilon_closure(self.q0)
        for symbol in input_string:
            if symbol not in self.T:
                return False
            next_states = set()
            for state in current_states:
                symbol_states = self.get_next_Q(state, symbol)
                next_states.update(symbol_states)
            current_states = set()
            for state in next_states:
                current_states.update(self.get_epsilon_closure(state))
            if not current_states:
                return False
        return any(state in self.F for state in current_states)

    def to_nfa(self):
        nfa = NFA(set(), self.T.copy(), self.q0, set())
        for state in self.Q:
            epsilon_closure = self.get_epsilon_closure(state)
            for symbol in self.T:
                next_states = set()
                for epsilon_state in epsilon_closure:
                    direct_states = self.get_next_Q(epsilon_state, symbol)
                    for direct_state in direct_states:
                        next_states.update(self.get_epsilon_closure(direct_state))
                if next_states:
                    nfa.add_transition(state, symbol, next_states)
            if any(s in self.F for s in epsilon_closure):
                nfa.F.add(state)
        nfa.Q.update(self.Q)
        return nfa

    def _add_transitions_to_graph(self, dot, state_to_node_id):
        transitions_map = {}
        for (state, symbol), next_states in self.transitions.items():
            for next_state in next_states:
                key = (state, next_state)
                if key not in transitions_map:
                    transitions_map[key] = []
                label = 'ε' if symbol == self.EPSILON else symbol
                transitions_map[key].append(label)
        for (state, next_state), symbols in transitions_map.items():
            label = ', '.join(symbols)
            dot.edge(state_to_node_id[state], state_to_node_id[next_state], label=label)

class NFA(EpsilonNFA):
    """
    非确定性有限自动机 (NFA)。
    可以看作是没有 Epsilon 转移的 EpsilonNFA。
    """
    def add_transition(self, from_state, symbol, to_Q):
        if symbol == self.EPSILON:
            raise ValueError("NFA cannot have Epsilon transitions. Use EpsilonNFA class instead.")
        super().add_transition(from_state, symbol, to_Q)

    def get_epsilon_closure(self, state_or_Q):
        if isinstance(state_or_Q, set):
            return state_or_Q.copy()
        else:
            return {state_or_Q}

    def accepts(self, input_string):
        return super().accepts(input_string)

    def to_dfa(self):
        dfa_initial = frozenset({self.q0})
        dfa_states = {dfa_initial}
        dfa_accept = set()
        unprocessed = [dfa_initial]
        dfa = DFA(set(), self.T.copy(), dfa_initial, set())
        while unprocessed:
            current_dfa_state = unprocessed.pop(0)
            for symbol in self.T:
                next_nfa_states = set()
                for nfa_state in current_dfa_state:
                    next_nfa_states.update(self.get_next_Q(nfa_state, symbol))
                if next_nfa_states:
                    next_dfa_state = frozenset(next_nfa_states)
                    dfa.add_transition(current_dfa_state, symbol, next_dfa_state)
                    if next_dfa_state not in dfa_states:
                        dfa_states.add(next_dfa_state)
                        unprocessed.append(next_dfa_state)
                else:
                    empty_state = frozenset()
                    dfa.add_transition(current_dfa_state, symbol, empty_state)
                    if empty_state not in dfa_states:
                        dfa_states.add(empty_state)
                        for sym in self.T:
                            dfa.add_transition(empty_state, sym, empty_state)
        for dfa_state in dfa_states:
            if any(nfa_state in self.F for nfa_state in dfa_state):
                dfa.F.add(dfa_state)
        dfa.Q = dfa_states
        return dfa

class DFA(Automaton):
    """
    确定性有限自动机 (DFA)。
    转移函数格式: {(state, symbol): single_next_state}
    """
    def _initialize_transitions(self):
        return {}

    def add_transition(self, from_state, symbol, to_state):
        if symbol not in self.T:
            raise ValueError(f"Invalid symbol: {symbol}")
        key = (from_state, symbol)
        if key in self.transitions and self.transitions[key] != to_state:
            raise ValueError(f"Non-deterministic transition attempted for DFA: {key} -> {to_state} (already maps to {self.transitions[key]})")
        self.transitions[key] = to_state
        self.Q.add(from_state)
        self.Q.add(to_state)

    def get_next_Q(self, current_state, symbol):
        return self.transitions.get((current_state, symbol))

    def is_complete(self):
        for state in self.Q:
            for symbol in self.T:
                if (state, symbol) not in self.transitions:
                    return False
        return True

    def __repr__(self):
        def state_to_str(s):
            if isinstance(s, frozenset):
                return str(set(sorted(list(s)))) if s else '{}'
            return str(s)
        sorted_Q = sorted(list(self.Q), key=state_to_str)
        Q_str = "{" + ", ".join(map(state_to_str, sorted_Q)) + "}"
        sorted_F = sorted(list(self.F), key=state_to_str)
        F_str = "{" + ", ".join(map(state_to_str, sorted_F)) + "}"
        q0_str = state_to_str(self.q0)
        trans_str = "{\n"
        sorted_transitions = sorted(self.transitions.items(), key=lambda item: (state_to_str(item[0][0]), item[0][1]))
        for (state, symbol), next_state in sorted_transitions:
            trans_str += f"    ({state_to_str(state)}, '{symbol}'): {state_to_str(next_state)},\n"
        trans_str += "  }"
        return (f"{type(self).__name__}(\n"
                f"  Q={Q_str},\n"
                f"  T={sorted(list(self.T))},\n"
                f"  transitions={trans_str},\n"
                f"  q0={q0_str},\n"
                f"  F={F_str}\n"
                f")")

    def accepts(self, input_string):
        current_state = self.q0
        for symbol in input_string:
            if symbol not in self.T:
                return False
            current_state = self.get_next_Q(current_state, symbol)
            if current_state is None:
                return False
        return current_state in self.F

    def _add_transitions_to_graph(self, dot, state_to_node_id):
        transitions_map = {}
        for (state, symbol), next_state in self.transitions.items():
            if (isinstance(state, frozenset) and not state) or (isinstance(next_state, frozenset) and not next_state):
                continue
            key = (state, next_state)
            if key not in transitions_map:
                transitions_map[key] = []
            transitions_map[key].append(symbol)
        for (state, next_state), symbols in transitions_map.items():
            label = ', '.join(symbols)
            dot.edge(state_to_node_id[state], state_to_node_id[next_state], label=label)

def main():

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
        print("3. 查看自动机")
        print("4. 保存自动机")
        print("5. 可视化自动机")
        print("6. 返回主菜单")
        
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
        
        elif choice == '3':
            print("\n当前自动机:")
            print(automaton)
        
        elif choice == '4':
            try:
                filename = input("输入保存文件名: ").strip()
                automaton.save(filename)
                print(f"自动机已保存到文件 {filename}")
            except Exception as e:
                print(f"保存失败: {e}")
        
        elif choice == '5':
            try:
                filename = input("输入可视化输出文件名 (可选): ").strip()
                automaton.visualize(filename if filename else None)
                print("可视化完成。")
            except Exception as e:
                print(f"可视化失败: {e}")
        
        elif choice == '6':
            break
        
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()


