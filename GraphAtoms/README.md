# 新的代码设计思路——数据和操作隔离

类只表示数据(或可进行简单操作)，函数只进行(复杂)操作

## geometry

- [ ] neighborlist(Z, R, cell, pbc) -> ijSDd
- [ ] bondlist(Z, i, j, d) -> ijdo
- [ ] atom_inner_outer

## dataclasses

- [ ] 使用pydantic和numpydantic构建基础类
- [ ] 基础类支持dict、str、bytes的快速双向转换
- [ ] 基础类支持json、yaml、toml、pickle、npz的快速双向转换（主要依赖于与dict的转换）
- [ ] 在基础类之上构建相应的类
  - [ ] Box(cell+pbc)
  - [ ] Energitics(E,Fmax, Freqs)
  - [ ] Matter(Z)
  - [ ] Structure(Matter, Box, Energitics, R)
    - [ ] 增加与ase.Atoms的双向转换
    - [ ] 增加与pymatgen.Structure的双向转换
  - [ ] BondGraph(Matter, i,j,o)
    - [ ] 增加与rdmol的双向转换
    - [ ] 增加与igraph的双向转换
    - [ ] 增加与networkx的双向转换
    - [ ] 增加与rustworkx的双向转换
    - [ ] 增加与pyg.Data的双向转换
  - [ ] SysGraph(Structure, BondGraph)
  - [ ] SysCluster 【最低优先度】
  - [ ] SysGas

## reaction

KMC中的Event与MC中的Move的共同之处就在于，它们都可以使得初始结构产生一个变化。不同之处在于KMC Event的接受概率为1。

- [ ] 构造结构变化类，该类是一个可调用对象，即存在call函数且返回值为一个Structure和相应的接受概率
- [ ] 基于结构变化类构造各种MC Move
  - [ ] 单粒子扰动
  - [ ] 粒子交换
  - [ ] 多粒子扰动
  - [ ] 等等
- [ ] 基于结构变化类构造Event类，该类存在R、T、G、P四个属性





