import pandas as pd
import numpy as np

# 读取表格文件
df = pd.read_csv('a.csv')
df['CT'] = pd.to_numeric(df['CT'], errors='coerce')
# 假设CT列在表格中
ct_column = df['CT']

# 新建一个列表存储处理后的CT值
new_ct_values = []

# 对CT列的数据进行处理
for i in range(0, len(ct_column), 3):
    # 每三个数据为一组
    if i + 2 < len(ct_column):
        # 取出当前的三个数据
        data1 = ct_column[i]
        data2 = ct_column[i + 1]
        data3 = ct_column[i + 2]
        
        # 计算两两之间的差值
        diff1 = abs(data1 - data2)
        diff2 = abs(data2 - data3)
        diff3 = abs(data1 - data3)
        
        # 判断差值是否都大于1
        if diff1 > 1 and diff2 > 1 and diff3 > 1:
            # 如果都大于1，则不保留这三个数
            new_ct_values.extend([None, None, None])
        else:
            if diff1 < 1 and diff2 < 1 and diff3 < 1:
                new_value=(data1 + data2+data3) / 3
            else:
            # 否则，找出最小的两个差值，计算对应的均值
                min_diff = min(diff1, diff2, diff3)
                
                if diff1 == min_diff:
                    # data1, data2留下来
                    new_value = (data1 + data2) / 2
                elif diff2 == min_diff:
                    # data2, data3留下来
                    new_value = (data2 + data3) / 2
                else:
                    # data1, data3留下来
                    new_value = (data1 + data3) / 2
                
                # 将计算得到的均值加入新的CT值列表
            new_ct_values.extend([new_value, new_value, new_value])
    
    else:
        # 处理剩余不足三个的情况，直接加入到新的CT值列表中
        new_ct_values.extend(ct_column[i:])

# 创建一个新的DataFrame来存储结果
result_df = df.copy()

# 将新计算的CT值替换原来的CT列
result_df['CT'] = new_ct_values
result_df = result_df.dropna(subset=['CT'])
# 删除含有None的行
# result_df = result_df.dropna()

# 保存处理后的表格文件
result_df.to_csv('output_file.csv', index=False, encoding='utf-8')
df_intermediate = pd.read_csv('output_file.csv')

# 每三行取第一行
final_df = df_intermediate.iloc[::3, :]

# 保存最终结果
final_df.to_csv('final_output0.csv', index=False, encoding='utf-8')

print("处理完成，已将每三行中的第一行写入 final_output.csv")






df = pd.read_csv('final_output0.csv')  # 假设文件是以制表符分隔的

# 提取Target Name列为ACTB的行
actb_rows = df[df['Target Name'] == 'actb']

# 创建一个字典以便快速查找每个Sample Name对应的ACTB行的CT值
actb_ct_dict = {row['Sample Name']: row['CT'] for _, row in actb_rows.iterrows()}

# 创建一个新的列以存储计算后的CT值差
df['CT Diff'] = None

# 遍历所有行，计算CT值差
for index, row in df.iterrows():
    if row['Target Name'] != 'ACTB':
        sample_name = row['Sample Name']
        if sample_name in actb_ct_dict:
            actb_ct = actb_ct_dict[sample_name]
            df.at[index, 'CT Diff'] = row['CT'] - actb_ct

# 保存处理后的表格文件
df.to_csv('final_output1.csv', index=False, encoding='utf-8')

print("处理完成，结果已写入 final_output1.csv")




target_names = df['Target Name'].unique()

# 创建一个新的列以存储计算后的CT值差
df['(E-C)CT Diff'] = None
df['2^-CT Diff'] = None
# 遍历所有Target Name
for target in target_names:
    if target == 'ACTB':
        continue
    # 过滤出当前Target Name的所有行
    target_df = df[df['Target Name'] == target]
    
    # 遍历E1到E6
    for i in range(1, 7):
        e_sample = f'E{i}-'
        c_sample = f'C{i}-'
        
        # 获取E行和C行
        e_row = target_df[target_df['Sample Name'].str.startswith(e_sample)]
        c_row = target_df[target_df['Sample Name'].str.startswith(c_sample)]
        
        # 如果E行和C行都存在，计算CT差值
        if not e_row.empty and not c_row.empty:
            e_index = e_row.index[0]
            c_index = c_row.index[0]
            
            e_ct = df.at[e_index, 'CT Diff']
            c_ct = df.at[c_index, 'CT Diff']
            if e_ct!=None and c_ct!=None:
                # 计算差值并保存到E行的CT Diff列
                df.at[e_index, '(E-C)CT Diff'] = e_ct - c_ct
                ct_diff=e_ct - c_ct
                df.at[e_index, '2^-CT Diff'] = 2 ** (-ct_diff)

# 保存处理后的表格文件
df.to_csv('2024-06-18 201454.csv', index=False, encoding='utf-8')

print("处理完成，结果已写入 final.csv")