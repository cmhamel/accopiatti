#pragma once
// #include <accopiatti/FunctionSpace.hpp>
#include <Panzer_STKConnManager.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>
#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_DOFManager.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

template<Scalar T>
class BasisFactory {
public:
    static Teuchos::RCP<Basis<T>> create(
        const stk::topology& topology,
        const FunctionSpaceType& function_space,
        const int& order
    ) {
        switch (function_space) {
        case FunctionSpaceType::Hcurl:
            throw std::runtime_error("Finish Hcurl wrapper methods");
        case FunctionSpaceType::Hdiv:
            throw std::runtime_error("Finish Hdiv wrapper methods");
        case FunctionSpaceType::Hgrad:
            return create_h1(topology, order);
        case FunctionSpaceType::L2:
            throw std::runtime_error("Finish L2 wrapper methods");
        }

        throw std::runtime_error("Unsupported feature encounted.");
    }

    // should we always be pinging the jacobian
    // off of linear shpae functions?
    // probably a weakness for curved...
    static Teuchos::RCP<Basis<T>> create_geometry(const stk::topology& topology) {
        switch (topology) {
        // case stk::topology::QUADRILATERAL_4_2D:
        //     return Teuchos::rcp(new HgradQuadBasis<T>(2));
        // case stk::topology::HEXAHEDRON_8:
        //     return Teuchos::rcp(new HgradHexBasis<T>(2));
        case stk::topology::QUADRILATERAL_4_2D:
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<Device, T, T>());
        case stk::topology::HEXAHEDRON_8:
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<Device, T, T>());
        default:
            throw std::runtime_error("Unsupported topology for Hgrad basis: " + topology.name());
        }
    }
private:
    static Teuchos::RCP<Basis<T>> create_h1(const stk::topology& topology, const int& order) {
        switch (topology) {
        // case stk::topology::QUADRILATERAL_4_2D:
        //     return Teuchos::rcp(new HgradQuadBasis<T>(order));
        // case stk::topology::HEXAHEDRON_8:
        //     return Teuchos::rcp(new HgradHexBasis<T>(order));
        case stk::topology::QUADRILATERAL_4_2D:
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<Device, T, T>());
        case stk::topology::HEXAHEDRON_8:
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<Device, T, T>());
        default:
            throw std::runtime_error("Unsupported topology for Hgrad basis: " + topology.name());
        }

    }

    // TODO implement hcurl, hdiv, L2 wrappers
};

template<Scalar T, int Dim>
class BasisDofOrdering {
public:
    static std::map<int, int> create(const Teuchos::RCP<Basis<T>>& basis, const stk::topology& topology) {
        switch (topology) {
        case stk::topology::QUADRILATERAL_4_2D:
            auto c1_basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<Device, T, T>());
            auto cn_basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<Device, T, T>(1));
            auto dof_coords_1 = Kokkos::DynRankView<T>("dof_coords", c1_basis->getCardinality(), Dim);
            auto dof_coords_n = Kokkos::DynRankView<T>("dof_coords", cn_basis->getCardinality(), Dim);

            c1_basis->getDofCoords(dof_coords_1);
            cn_basis->getDofCoords(dof_coords_n);

            for (int i = 0; i < 4; ++i) {
                std::cout << "[" << i << "] = " << dof_coords_1(i, 0) << ", " << dof_coords_1(i, 1) << std::endl;
            }
            std::cout << std::endl;
            for (int i = 0; i < 4; ++i) {
                std::cout << "[" << i << "] = " << dof_coords_n(i, 0) << ", " << dof_coords_n(i, 1) << std::endl;
            }
            // break;
            auto dof_map = std::map<int, int>({
                {0, 0},
                {1, 1},
                {2, 3},
                {3, 2}
            });

            return dof_map;

        // default:
        //     throw std::runtime_error("Unsupported topology for Hgrad basis: " + topology.name());
        }
        throw std::runtime_error("Unsupported topology for Hgrad basis: " + topology.name());
    }
};

struct FieldSetupHelper {
    int basis_order;
    FieldType field_type;
    FunctionSpaceType function_space;
    int integration_order;
};

template<Scalar T, int Dim>
class DofManager {
public:
    explicit DofManager(
        stk::ParallelMachine comm,
        Mesh& mesh,
        const std::map<std::string, FieldSetupHelper>& field_helpers
    ) {
        dof_manager = Teuchos::rcp(new panzer::DOFManager);
        dof_manager->setConnManager(mesh.create_conn_manager(), comm);
        const auto block_tops = mesh.get_element_block_topologies();
        for (const auto& [field_name, field_helper] : field_helpers) {
            sol_field_names.push_back(field_name);
            for (const auto& [block_name, topology] : block_tops) {
                add_field(mesh, Dim, block_name, field_name, topology, field_helper);
            }
        }
        mesh.summarize(std::cout);
        dof_manager->buildGlobalUnknowns();
        std::cout << "Finished dof manager setup" << std::endl;
    }

    Kokkos::DynRankView<double, Device> get_dof_coordinates(const std::string& block_name) const {
        const auto& geom_basis = geom_basis_map.at(block_name);
        stk::mesh::EntityIdVector local_elem_ids;
        Kokkos::DynRankView<T, Device> points;
        panzer::Intrepid2FieldPattern coord_provider(geom_basis);
        auto conn_manager = dof_manager->getConnManager();
        auto stk_conn_manager = Teuchos::rcp_dynamic_cast<const panzer_stk::STKConnManager>(conn_manager);

        TEUCHOS_TEST_FOR_EXCEPTION(
            stk_conn_manager.is_null(),
            std::runtime_error,
            "DofManager::get_dof_coordinates: "
            "DOFManager does not contain an STKConnManager."
        );

        stk_conn_manager->getDofCoords(
            block_name,
            coord_provider,
            local_elem_ids,
            points
        );
        return points;
    }

    void get_element_global_dof_ids(const std::string& block_name, const LocalOrdinal& e, std::vector<GlobalOrdinal>& gids) {
        dof_manager->getElementGIDs(e, gids, block_name);
    }

    int get_field_id(const std::string& field_name) const {
        return dof_manager->getFieldNum(field_name);
    }

    void get_field_offsets(std::vector<GlobalOrdinal>& indices, std::string block_name, int field_num) const {
        dof_manager->getGIDFieldOffsets(block_name, field_num);
    }

    const auto& get_field_offsets_closure(
        const std::string& block_name,
        const std::string& field_name,
        const int& subcell_dim,
        const int& subcell_ord
    ) const {
        return dof_manager->getGIDFieldOffsets_closure(block_name, get_field_id(field_name), subcell_dim, subcell_ord);
    }

    // stk::mesh::EntityIdVector get_local_cell_ids(
    //     const std::string& block_name
    // ) const {
    //     stk::mesh::EntityIdVector local_cell_ids;

    //     const auto& part = get_element_block_part(block_name);

    //     const auto& bulk = get_bulk();

    //     const stk::mesh::BucketVector& buckets =
    //         bulk->get_buckets(
    //             stk::topology::ELEM_RANK,
    //             part
    //         );

    //     for (const stk::mesh::Bucket* bucket : buckets) {
    //         for (size_t i = 0; i < bucket->size(); ++i) {
    //             local_cell_ids.push_back(
    //                 bulk->identifier((*bucket)[i])
    //             );
    //         }
    //     }

    //     return local_cell_ids;
    // }

    int get_num_solution_fields() const {
        return sol_field_names.size();
    }

    void get_owned_and_ghosted_indices(std::vector<GlobalOrdinal>& indices) const {
        dof_manager->getOwnedAndGhostedIndices(indices);
    }

    void get_owned_indices(std::vector<GlobalOrdinal>& indices) const {
        dof_manager->getOwnedIndices(indices);
    }

    auto get_solution_basis(std::string block_name, std::string field_name) {
        return sol_basis_map[block_name][field_name];
    }

    auto get_solution_field_names() const {
        return sol_field_names;
    }

    void summarize(std::ostream& os) const {
        dof_manager->printFieldInformation(os);
    }

private:
    void add_field(
        Mesh& mesh,
        const int& spatial_dimension,
        const std::string& block_name,
        const std::string& field_name,
        stk::topology topology,
        const FieldSetupHelper& fspace_helper
    ) {
        auto geom_basis = BasisFactory<T>::create_geometry(topology);
        auto sol_basis = BasisFactory<T>::create(
            topology,
            fspace_helper.function_space,
            fspace_helper.basis_order
        );
        geom_basis_map[block_name] = geom_basis;
        sol_basis_map[block_name].emplace(field_name, sol_basis);

        // dof id map for consistency between stk and intrepid2 dof ordering
        stk_to_intrepid2_dof_maps[block_name] = BasisDofOrdering<T, Dim>::create(sol_basis, topology);

        auto pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(sol_basis));

        if (fspace_helper.field_type == FieldType::Scalar) {
            dof_manager->addField(block_name, field_name, pattern);
            mesh.add_solution_field(field_name, block_name);
        } else if (fspace_helper.field_type == FieldType::Vector) {
            std::vector<std::string> exts{"_x", "_y", "_z"};
            for (int i = 0; i < spatial_dimension; ++i) {
                dof_manager->addField(block_name, field_name + exts[i], pattern);
                mesh.add_solution_field(field_name + exts[i], block_name);
            }
        } else {
            throw std::runtime_error("Unsupported field type");
        }
    }

public:
    Teuchos::RCP<panzer::DOFManager> dof_manager;
    std::map<std::string, Teuchos::RCP<Basis<T>>> geom_basis_map;
    std::map<std::string, std::map<std::string, Teuchos::RCP<Basis<T>>>> sol_basis_map;
    std::vector<std::string> sol_field_names;
    std::map<std::string, std::map<int, int>> stk_to_intrepid2_dof_maps;
};

} // end namespace accopiatti
