import Pkg
Pkg.activate("Packmol", shared=true)
using Packmol
root = ENV["REPLICA_ROOT"]
for name in ["R01", "R02", "R03", "R04"]
    cd(joinpath(root, name)) do
        isfile("packed.pdb") && error("Refusing to replace existing packed.pdb")
        run_packmol("packmol.inp")
    end
end
