
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language(),name='python')

parser = Parser()
parser.set_language(PY_LANGUAGE)

active = False
active0 = False
break_ev = False
def remove_comments(code, large_size= -100000):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    global active
    global active0
    active = True
    active0 = False
    def traverse_and_collect_comments(node):
        global active
        global active0
        comments = []
        if node.type == 'comment':
            comments.append(node)
        if active0 == False and node.type == 'block':
            active0 = True
        if active0 ==True and active == False and node.type == 'string' and abs((node.end_byte - node.start_byte) - large_size) <= 1:
            print("maybe there is an error here")
            print(code[node.start_byte: node.end_byte])
            global break_ev
            break_ev = True
        if active0 == True and node.type == 'string' and active == True:
            active = False
            comments.append(node)

        for child in node.children:
            comments.extend(traverse_and_collect_comments(child))
        return comments

    comment_nodes = traverse_and_collect_comments(root_node)

    new_code_parts = []
    last_idx = 0

    for comment in comment_nodes:
        start_byte = comment.start_byte
        end_byte = comment.end_byte
        new_code_parts.append(code[last_idx:start_byte])
        last_idx = end_byte

    new_code_parts.append(code[last_idx:])
    return ''.join(new_code_parts)
def getbreak():
    global break_ev
    return break_ev