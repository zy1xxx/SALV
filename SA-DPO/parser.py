import re
import subprocess
import copy

class ASTNode:
    def __init__(self, nodetype, location=None, info=None, indent=0):
        self.nodetype = nodetype
        self.location = location
        self.info = info or {}
        self.indent = indent
        self.children = []
        self.parent = None
    def add_child(self, child):
        self.children.append(child)
    
    def __repr__(self):
        return f"{self.nodetype} {self.location} {self.info}"

class ASTTree:
    """
    ASTTree parser to construct ast trees for each module.
    getting related lines for certain signal.
    Attributes:
        code_text (Str): code text.
        module_tree (Dict): all ast trees, each ast tree represents a module.
        main_module_name (Str): top module name.
    """

    def __init__(self,ast_text,code_text):
        self.code_text=code_text
        self.code_lines=self.code_text.split("\n")
        self.module_tree = self.parse_ast_for_all_modules(ast_text)
        self.main_module_name=next(iter(self.module_tree))        
        self.related_lines_str=""

    def get_related_lines(self,ouput_signal,module_name=None,recursion=True,include_io=False,exclude_io=False):
        """
        get related_lines,related_blocks and related signals for a certain signal

        Args:
            ouput_signal (Str): signal name for query.
            module_name (Str): module name for query. if not provided, use the top module instead.
            recursion (Bool): if query sub module recursively.
            include_io (Bool):  
            exclude_io (Bool):
        Returns:
            related_lines (Set): related Assignment lines, including blocking assignment and Non-blocking assignment.
            related_blocks (Set): related Blocks, such as always block.
            related_names (Set): related signal names
        
        return: List,List
        """

        #相关block行
        related_names=set()
        related_lines=set()
        related_block_lines=set()

        wait_traversed_names={ouput_signal}#遍历的信号名字
        already_traversed_names=set()#已经遍历过的信号名字
        
        if module_name:
            if module_name in self.module_tree.keys():
                ast_tree=self.module_tree[module_name]["module_node"]
            else:
                #异常没有相关子模块,返回空的
                return related_lines,related_block_lines,related_names
        else:
            ast_tree=list(self.module_tree.values())[0]["module_node"]
            module_name=self.main_module_name
        
        def get_related_module_head_block_lines(ast_tree,module_name):
            module_name=module_name.strip("\\")
            #获取module head和end的lines
            related_module_head_block_lines=set()
            pattern = rf"module\s+{re.escape(module_name)}\s*(#\s*\([\s\S]+\))?\s*(//.*\s)?\([\s\S]+?\);"
            match = re.search(pattern, self.code_text)
            if match:
                module_head = match.group(0)
                module_head_line_ctn=len(module_head.split("\n"))
                module_start=ast_tree.children[0].location[0][0]
                module_end=ast_tree.children[0].location[1][0]
                related_module_head_block_lines.add((module_start,module_start+module_head_line_ctn-1))
                related_module_head_block_lines.add((module_end,module_end))
            return related_module_head_block_lines
        
        def get_identifer_names(node):
            #获取一个节点的identifer_names
            identifer_names=[]
            def traverse(node):
                if node.nodetype=='AST_IDENTIFIER':
                    if 'str' in node.info.keys():
                        identifer_names.append(node.info['str'])
                else:
                    for child_node in node.children:
                        traverse(child_node)
            traverse(node)
            return identifer_names
            
        def get_condition_related_signals(assign_node):
            #获取assign_node的相关信号名称
            related_signals=[]
            node=assign_node
            while node.parent:
                if node.nodetype=='AST_CASE':
                    if len(node.children)>=1:
                        identifer_names=get_identifer_names(node.children[0])
                        related_signals.extend(identifer_names)
                elif node.nodetype=='AST_ALWAYS':
                    for child_node in node.children:
                        if child_node.nodetype=='AST_POSEDGE' or child_node.nodetype=='AST_NEGEDGE':
                            identifer_names=get_identifer_names(child_node)
                            related_signals.extend(identifer_names)
                node=node.parent
            return set(related_signals)
                        
        def get_L_and_R_value_name(assign_node):
            #获取assign_node的左值和右值信号名称
            L_value_names=[]
            R_value_names=[]
            def traverse_assign_node(node):
                if node:
                    if node.nodetype=='AST_IDENTIFIER':
                        if 'str' in node.info.keys():
                            if 'type' in node.info.keys() and node.info['type']=='in_lvalue':
                                L_value_names.append(node.info['str'])
                            else:
                                R_value_names.append(node.info['str'])
                    else:
                        for child_node in node.children:
                            traverse_assign_node(child_node)
            traverse_assign_node(assign_node)
            return L_value_names,R_value_names
        
        def get_related_block_node(node):
            #获取相关的always节点
            while node.parent:
                if node.nodetype=='AST_ALWAYS':
                    return node
                node=node.parent
            return None

        def get_L_and_R_value_names_in_module_call(node):
            #判断信号是否是Lvalue,在module调用中,并返回对应的module_class_name,module_port_name
            module_class_name=""
            port_id=0
            L_value_names={}
            R_value_names={}

            for child in node.children:
                if child.nodetype=="AST_CELLTYPE":
                    if "str" in child.info.keys():
                        module_class_name=child.info["str"]
                    else:
                        #异常情况，没有调用module的名称
                        return L_value_names,R_value_names,module_class_name 
                elif child.nodetype=="AST_ARGUMENT":
                    if module_class_name in self.module_tree.keys():
                        if "str" in child.info.keys():
                            sub_module_port_name=child.info["str"]
                        else:
                            sub_module_port_names=list(self.module_tree[module_class_name]["port_info"].keys())
                            if port_id< len(sub_module_port_names):
                                sub_module_port_name=sub_module_port_names[port_id]
                            else:
                                #异常情况，调用的参数多于定义的参数,直接返回
                                return L_value_names,R_value_names,module_class_name
                        
                        port_id+=1
                        port_type=self.module_tree[module_class_name]["port_info"][sub_module_port_name]["port_type"]
                        if port_type=="output":
                            L_value_names[sub_module_port_name]=get_identifer_names(child)
                        elif port_type=="input":
                            R_value_names[sub_module_port_name]=get_identifer_names(child)

            return L_value_names,R_value_names,module_class_name
        def traverse_ast(node,search_name,related_names,related_lines,related_block_lines,include_io,exclude_io):
            for node in node.children:
                if node:
                    if node.nodetype=='AST_ASSIGN_LE' or node.nodetype=="AST_ASSIGN" or node.nodetype=='AST_ASSIGN_EQ': #赋值语句
                        #node是一个赋值语句
                        L_value_names,R_value_names=get_L_and_R_value_name(node)
                        if search_name in L_value_names:
                            #查找到了
                            related_lines.add((node.location[0][0],node.location[1][0])) #添加相关行
                            
                            related_names.update(set(R_value_names)) #添加赋值相关信号
                            related_names.update(get_condition_related_signals(node)) # 添加条件相关信号    
                            
                            related_block_node=get_related_block_node(node) #添加相关always块
                            if related_block_node and related_block_node.location:
                                related_block_lines.add((related_block_node.location[0][0],related_block_node.location[1][0]))
                    elif node.nodetype=='AST_WIRE': #定义语句
                        if 'str' in node.info.keys():
                            if 'type' in node.info.keys():
                                if node.info['str']==search_name:
                                    if not exclude_io and (node.info['type']=='input' or node.info['type']=='output'):
                                        related_lines.add((node.location[0][0],node.location[1][0]))
                                else:
                                    if include_io and (node.info['type']=='input' or node.info['type']=='output'):
                                        related_lines.add((node.location[0][0],node.location[1][0]))
                            else:
                                if node.info['str']==search_name:
                                    related_lines.add((node.location[0][0],node.location[1][0]))
                    elif node.nodetype=='AST_CELL': #调用模块
                        L_value_names,R_value_names,module_class_name=get_L_and_R_value_names_in_module_call(node) 
                        for sub_module_port_name,signal_names in L_value_names.items():
                            if search_name in signal_names:
                                related_lines.add((node.location[0][0],node.location[1][0]))
                                if recursion:
                                    if module_class_name=="":
                                        continue
                                    if module_class_name != module_name:#
                                        #搜索信号是子模块的output，则递归寻找子模块相关行
                                        sub_module_related_lines,sub_module_related_block_lines,sub_module_related_names=self.get_related_lines(sub_module_port_name,module_class_name,recursion=recursion,include_io=include_io,exclude_io=exclude_io)                       
                                        related_lines.update(sub_module_related_lines)
                                        related_block_lines.update(sub_module_related_block_lines)
                                        #将条件信号加入到相关信号中
                                        for sub_module_port_name,signal_names in R_value_names.items():
                                            if sub_module_port_name in sub_module_related_names:
                                                related_names.update(set(signal_names))
                    traverse_ast(node,search_name,related_names,related_lines,related_block_lines,include_io,exclude_io)
        
        #没有等待遍历的信号退出
        while len(wait_traversed_names)!=0:
            search_name = wait_traversed_names.pop()#取出一个需要遍历的中间信号
            traverse_ast(ast_tree,search_name,related_names,related_lines,related_block_lines,include_io,exclude_io)#从根节点遍历该中间信号: 1.找出赋值语句 2.找出相关信号
            already_traversed_names.add(search_name)#将该中间信号加入已经遍历的信号集合中
            for name in related_names:#将该中间信号的相关信号加入到未遍历集合中
                if name not in already_traversed_names:
                    wait_traversed_names.add(name)
        
        ##module head和module end作为相关block
        related_block_lines.update(get_related_module_head_block_lines(ast_tree,module_name))

        return related_lines,related_block_lines,related_names
    
    def subtract_lines(self,A_lines,B_lines):
        """
        substract B_lines from A_lines
        """
        merged_lines=copy.deepcopy(A_lines)
        for B_line in B_lines:
            B_start=B_line[0]
            B_end=B_line[1]
            # print("log:",merged_lines,B_line)
            for A_line in merged_lines:
                A_start=A_line[0]
                A_end=A_line[1]
                if not (B_end<A_start or B_start>A_end):
                    merged_lines.remove(A_line)
                    if B_start>A_start:
                        merged_lines.append((A_start,B_start-1))
                    if B_end<A_end:
                        merged_lines.append((B_end+1,A_end))
        merged_lines.sort(key=lambda x: (x[0],x[1]))
        return merged_lines

    def merge_lines(self,A_lines,B_lines):
        """
        add B_lines into A_lines
        """
        merged_lines=copy.deepcopy(A_lines)
        for B_line in B_lines:
            B_start=B_line[0]
            B_end=B_line[1]

            in_block=False
            for A_line in A_lines:
                A_start=A_line[0]
                A_end=A_line[1]
                if B_start >= A_start and B_end <= A_end:
                    in_block=True
                    break
            if not in_block:
                merged_lines.append(B_line)
                    
        merged_lines.sort(key=lambda x: (x[0],x[1]))
        return merged_lines
    
    def parse_ports(self,root):
        """
        parse module ports 

        Args:
            root (Str): root node for a module
            port_info (Dict): key:port_name, value:{"port_type":port_type,"port_id":port_id}
        Returns:

        """
        module_root=root.children[0]
        port_info={}
        for child in module_root.children:
            if child.nodetype=="AST_WIRE":
                if "type" in child.info and "port" in child.info:
                    port_type=child.info["type"]
                    port_name=child.info["str"]
                    port_id=child.info["port"]
                    port_info[port_name]={"port_type":port_type,"port_id":int(port_id)}
        sorted_port_info = dict(sorted(port_info.items(), key=lambda item: item[1]["port_id"]))
        return sorted_port_info
    
    def parse_ast_for_all_modules(self,ast_text):
        """
        parse all modules 
        
        Args:
            ast_text (Str): ast_text from yosys

        Returns:
            module_tree (Dict): key:module_name, value:{"module_node":root_node,"port_info":port_info}
        """
        split_pattern="END OF AST DUMP"
        if split_pattern in ast_text:
            module_ast=ast_text.split(split_pattern)
        else:
            module_ast=[ast_text]

        module_tree={}
        for ast_text in module_ast:
            lines=ast_text.split("\n")
            node,module_name=self.parse_ast(lines)
            port_info=self.parse_ports(node)
            module_tree[module_name]={"module_node":node,"port_info":port_info}
        return module_tree

    def parse_ast(self,lines):
        """
        parse ast to construct a tree
        
        Args:
            lines (List): ast text lines.
        
        Returns:
            node (ASTNode): root node 
            module_name (Str): module name
        
        """
        node_stack = []
        root = ASTNode("ROOT")
        node_stack.append((root, -1))
        for line in lines:
            if not line.strip():
                continue
            
            # 计算缩进层级（每两个空格算一层）
            indent = len(line) - len(line.lstrip(' '))
            
            # 使用正则提取主要字段
            match = re.match(r'\s*(AST_\w+)\s+<([^>]+)>\s+\[([^\]]+)\](.*)', line)
            if not match:
                continue
    
            nodetype, location, pointer, rest = match.groups()
            #location
            location=location.split(":")[1]
            start_s=location.split('-')[0]
            end_s=location.split('-')[1]
            location_tuple=((int(start_s.split(".")[0]),int(start_s.split(".")[1])),(int(end_s.split(".")[0]),int(end_s.split(".")[1])))
            
            # 提取 info 字段，如 str='\name' port=1 等
            info = {}
            str_match = re.search(r"str='([^']+)'", rest)
            if str_match:
                info['str'] = str_match.group(1)
    
            port_match = re.search(r'port=(\d+)', rest)
            if port_match:
                info['port'] = int(port_match.group(1))
    
            type_match = re.search(r'(input|output|reg|in_lvalue)', rest)
            if type_match:
                info['type'] = type_match.group(1)
    
            node = ASTNode(nodetype, location=location_tuple, info=info, indent=indent)

            # 栈操作：寻找正确的父节点
            while node_stack and node_stack[-1][1] >= indent:
                node_stack.pop()
            parent_node = node_stack[-1][0]
            parent_node.add_child(node)
            node.parent=parent_node
            node_stack.append((node, indent))
        if root.children and root.children[0].nodetype=="AST_MODULE":
            return root,root.children[0].info['str']
        else:
            return root,"null"
    
    def get_signal_related_lines(self,ouput_signal,recursion=True,include_io=False,exclude_io=False):
        """
        get sorted related lines for a certain output signal

        Args:
            ouput_signal (Str): signal name for query.
            recursion (Bool): if query sub module recursively.
            include_io (Bool):  
            exclude_io (Bool):
        Returns:
            related_lines (List): related Assignment lines, including blocking assignment and Non-blocking assignment.
            related_blocks (List): related Blocks, such as always block.
        
        """
        related_lines,related_block_lines,related_names = self.get_related_lines(ouput_signal,module_name=None,recursion=recursion,include_io=include_io,exclude_io=exclude_io)
        #排序相关信号和block行
        related_lines=list(related_lines)
        related_lines.sort(key=lambda x: (x[0],x[1]))
        related_block_lines=list(related_block_lines)
        related_block_lines.sort(key=lambda x: (x[0],x[1]))

        return related_lines,related_block_lines
    
    def get_only_related_lines(self,c_signals,ic_signals,recursion=True,log_flag=False,keep_all_io=False):
        """
        get related lines for a certain output signal and exclude unrelated assignment and block lines.
        
        Args:
            c_signals (List): related output signal list
            ic_signals (List): unrelated output signal list
            recursion (Bool): if query sub module recursively. 
            log_flag (Bool): if visualize the extract code.
            keep_all_io (Bool):
        Returns:
            all_c_signal_only_lines (List): related lines for certain output signals.
        
        """

        all_c_signal_related_lines_merge=[]
        all_c_signal_related_lines=[] #
        for signal_name in c_signals:
            if keep_all_io:
                c_related_lines,c_related_block_lines=self.get_signal_related_lines(f"\\{signal_name}",recursion=recursion,include_io=True)
            else:
                c_related_lines,c_related_block_lines=self.get_signal_related_lines(f"\\{signal_name}",recursion=recursion)
            c_related_lines_merge=self.merge_lines(c_related_block_lines,c_related_lines)#合并得到相关的所有行
            all_c_signal_related_lines_merge=self.merge_lines(all_c_signal_related_lines_merge,c_related_lines_merge)#得到所有信号的所有行
            all_c_signal_related_lines=self.merge_lines(all_c_signal_related_lines,c_related_lines)#得到所有信号的赋值语句行
        
        all_c_signal_only_lines=copy.deepcopy(all_c_signal_related_lines_merge) #信号只相关的行
        for signal_name in ic_signals:
            if keep_all_io:
                i_related_lines,i_related_block_lines=self.get_signal_related_lines(f"\\{signal_name}",recursion=recursion,exclude_io=True)
            else:
                i_related_lines,i_related_block_lines=self.get_signal_related_lines(f"\\{signal_name}",recursion=recursion)
            i_only_lines=self.subtract_lines(i_related_lines,all_c_signal_related_lines)#不相关信号的赋值语句行-相关的所有行
            all_c_signal_only_lines=self.subtract_lines(all_c_signal_only_lines,i_only_lines)
        
        # log
        lines=self.code_lines
        if log_flag:
            for i in range(len(lines)):
                related_flag=False
                for c_only_line in all_c_signal_only_lines:
                    if i >=c_only_line[0]-1 and i <c_only_line[1]:
                        related_flag=True
                        break
                if related_flag:
                    print("\u2705"+lines[i])
                    self.related_lines_str+=lines[i]+"\n"
                else:
                    print("\u274C"+lines[i])
                    # if not lines[i].startswith("//"):
                    #     self.related_lines_str+="begin end\n"
                    # pass
        
        return all_c_signal_only_lines
    
    def get_select_token_index(self,tokenizer,response,signal_info):
        """
        get related tokens for a certain output signal and exclude unrelated assignment and block lines tokens.
        
        Args:
            tokenizer: tokenizer to use.
            response: whole reponse text.
            signal_info (Dict): {"correct_signals":[],"incorrect_signals":[]}
                "correct_signals": represents related output signals 
                "incorrect_signals": represents unrelated output signals
        Return:
            select_token_index_ls (List): [(start_index,end_index)]
        
        """
        all_c_signal_only_lines=self.get_only_related_lines(signal_info["correct_signals"],signal_info["incorrect_signals"],recursion=True)
        offset = response.find(self.code_text)
        code_lines=self.code_lines
        
        code_line_start=[]#保存code line的local字符index
        pt=0
        for line in code_lines:
            code_line_start.append(pt)
            pt=pt+len(line)+1#\n算一个
        
        select_char_index_ls=[]#line->char
        for line_index in all_c_signal_only_lines:
            start_index=line_index[0]
            end_index=line_index[1]
            if start_index<1:
                #为什么all_signal_c_only_lines会出现(0,0)
                continue
            elif end_index>=len(code_lines):
                select_char_index_ls.append((offset+code_line_start[start_index-1],offset+len(self.code_text)))
            else:
                select_char_index_ls.append((offset+code_line_start[start_index-1],offset+code_line_start[end_index]))
        
        other_char_index_ls=[]
        for sub_string in ["```verilog\n","\n```"]:
            if sub_string in response:
                other_char_index_ls.append((response.index(sub_string),response.index(sub_string)+len(sub_string)))
        select_char_index_ls=select_char_index_ls+other_char_index_ls
        
        select_char_index_ls=sorted(select_char_index_ls,key=lambda x:x[0])

        offsets=tokenizer(response,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']

        select_token_index_ls=[]
        for select_char_index in select_char_index_ls:
            # token_indices = [i for i, (start, end) in enumerate(offsets) if start >= select_char_index[0] and end <= select_char_index[1]]
            token_indices =[] 
            flag="Null"
            for i, (start, end) in enumerate(offsets):
                if flag=="Null":
                    if start <= select_char_index[0] and end> select_char_index[0]:
                        token_indices.append(i)
                        flag="started"
                    if start < select_char_index[1] and end >= select_char_index[1]:
                        flag="ended"
                elif flag=="started":
                    if start < select_char_index[1] and end >= select_char_index[1]:
                        token_indices.append(i)
                        flag="ended"
                    else:
                        #中间的token
                        token_indices.append(i)
                elif flag=="ended":
                    break
            
            if token_indices!=[]:
                select_token_index_ls.append((token_indices[0],token_indices[-1]+1))
            else:
                print(offsets)
                print(select_char_index)
                raise ValueError
        return select_token_index_ls

    def get_correct_signal_implement(self,signal_info,fixed_text="```verilog\n"):
        """
        Deprecated
        """
        all_c_signal_only_lines=self.get_only_related_lines(signal_info["correct_signals"],signal_info["incorrect_signals"],recursion=False)
        lines=self.code_lines
        for i in range(len(lines)):
            related_flag=False
            if "endmodule" in lines[i]:
                continue
            for c_only_line in all_c_signal_only_lines:
                if i >=c_only_line[0]-1 and i <c_only_line[1]:
                    related_flag=True
                    break
            if related_flag:
                fixed_text+=lines[i]
                fixed_text+="\n"
            else:
                pass
        return fixed_text

if __name__ == '__main__':
    code_file_name="single_module.v"
    code=open(code_file_name,"r").read()
    result=subprocess.run(f"yosys -p 'read_verilog -sv -dump_ast1 {code_file_name}'", shell=True,capture_output=True,text=True)
    stdout=result.stdout
    
    pattern="Dumping AST before simplification:(.*)--- END OF AST DUMP ---"
    match = re.search(pattern, stdout, re.DOTALL)
    if match:
        ast_text = match.group(1).strip()
    else:
        print("没找到匹配内容")
    
    # with open("ast_txt_log.txt","w") as f:
    #     f.write(ast_text)
    
    ast_tree=ASTTree(ast_text,code)
    # signal_info={"correct_signals":["d"],"incorrect_signals":["e","a","b","c","aluctr"]}
    signal_info={"correct_signals":["count"],"incorrect_signals":["carry","even_flag"]}

    # print(ast_tree.get_correct_signal_implement(signal_info))
    ast_tree.get_only_related_lines(signal_info["correct_signals"],signal_info["incorrect_signals"],recursion=True,log_flag=True,keep_all_io=True)
    
    # print(ast_tree.related_lines_str)
    
    # response=f"```verilog\n{code}\n```"
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("/share/collab/codemodel/models/Qwen/Qwen2.5-Coder-7B-Instruct")
    # token_indexes=ast_tree.get_select_token_index(tokenizer,response,signal_info)
    # print(token_indexes)
    # tokens=tokenizer(response,add_special_tokens=False,return_offsets_mapping=False)['input_ids']
    # # print(tokens)

    # for token_index in token_indexes:
    #     decoded_text = tokenizer.decode(tokens[token_index[0]:token_index[1]],skip_special_tokens=True)
    #     print(decoded_text,end="")
    
    

    

            
    
    