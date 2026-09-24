#include <accopiatti/DofManager.hpp>
#include <accopiatti/FunctionSpace.hpp>
#include <accopiatti/Mesh.hpp>

namespace accopiatti {

template<Scalar T, int Dim>
DofManager<T, Dim>::DofManager(
    stk::ParallelMachine comm,
    Mesh& mesh,
    const YAML::Node& solution_inputs
) {
    int spatial_dimension = mesh.get_dimension();
    dof_manager = Teuchos::rcp(new panzer::DOFManager);

    dof_manager->setConnManager(mesh.get_conn_manager(), comm);
    const auto block_tops = mesh.get_element_block_topologies();
    auto fields = parse_fields(solution_inputs);
    for (const auto& [field_name, field] : fields) {
        for (const auto& [block, topology] : block_tops) {
            FunctionSpace<T, Dim> fspace = FunctionSpace<T, Dim>(mesh, block, field, 1024);
            add_field(spatial_dimension, block, field_name, topology, field);
        }
    }

    dof_manager->buildGlobalUnknowns();
}

template<Scalar T, int Dim>
void DofManager<T, Dim>::add_field(
    const int& spatial_dimension,
    const std::string& block,
    const std::string& field_name,
    stk::topology topology,
    const FunctionSpaceHelper& field
) {
    auto basis = BasisFactory<T>::create(
        topology,
        field.function_space,
        field.basis_order
    );
    basis_map[block].emplace(field_name, basis);

    auto pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));

    if (field.field_type == FieldType::Scalar) {
        dof_manager->addField(block, field_name, pattern);
    } else if (field.field_type == FieldType::Vector) {
        std::vector<std::string> exts{"_x", "_y", "_z"};
        for (int i = 0; i < spatial_dimension; ++i) {
            dof_manager->addField(block, field_name + exts[i], pattern);
        }
    } else {
        throw std::runtime_error("Unsupported field type");
    }
}

template class DofManager<double, 1>;
template class DofManager<double, 2>;
template class DofManager<double, 3>;

} // end namespace accopiatti
