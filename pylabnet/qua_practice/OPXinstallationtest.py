from qm import QuantumMachinesManager
qmm = QuantumMachinesManager(host='192.168.88.252') # If you have multiple clusters in the network, use the main OPX of each cluster
qmm = QuantumMachinesManager(host='192.168.88.252', cluster_name='Cluster_1')
