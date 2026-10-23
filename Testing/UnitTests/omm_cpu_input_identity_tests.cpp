// CPU-only checks for the production importer identity and upload lifetime.
#include "../../Rtxpt/OpacityMicroMap/OmmCpuInputIdentity.h"
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace
{
    void Require(bool value, const char* message)
    {
        if (!value) { std::fprintf(stderr, "OMM CPU identity: %s\n", message); std::abort(); }
    }

    class FakeBuffer final : public nvrhi::RefCounter<nvrhi::IBuffer>
    {
    public:
        inline static int alive = 0;
        nvrhi::BufferDesc desc;
        explicit FakeBuffer(uint64_t size) { ++alive; desc.byteSize = size; }
        ~FakeBuffer() override { --alive; }
        const nvrhi::BufferDesc& getDesc() const override { return desc; }
        nvrhi::GpuVirtualAddress getGpuVirtualAddress() const override { return 0; }
    };

    std::shared_ptr<donut::engine::MeshInfo> Mesh()
    {
        auto mesh = std::make_shared<donut::engine::MeshInfo>();
        mesh->buffers = std::make_shared<donut::engine::BufferGroup>();
        mesh->buffers->indexData = {0, 1, 2};
        mesh->buffers->texcoord1Data = {{0,0}, {1,0}, {0,1}};
        mesh->totalIndices = mesh->totalVertices = 3;
        auto geometry = std::make_shared<donut::engine::MeshGeometry>();
        geometry->numIndices = geometry->numVertices = 3;
        mesh->geometries.push_back(geometry);
        return mesh;
    }

    void Upload(donut::engine::BufferGroup& buffers)
    {
        buffers.indexBuffer = nvrhi::BufferHandle::Create(new FakeBuffer(buffers.indexData.size() * 4));
        buffers.vertexBuffer = nvrhi::BufferHandle::Create(new FakeBuffer(128 + buffers.texcoord1Data.size() * 8));
        buffers.getVertexBufferRange(donut::engine::VertexAttribute::TexCoord1) =
            nvrhi::BufferRange(128, buffers.texcoord1Data.size() * 8);
        std::vector<uint32_t>().swap(buffers.indexData);
        std::vector<donut::math::float2>().swap(buffers.texcoord1Data);
    }

    void CheckExactEquality()
    {
        const std::array<uint32_t, 3> indices{0, 1, 2}, renumbered{2, 0, 1}, reordered{0, 2, 1};
        const std::array<donut::math::float2, 3> uv{{{0,0}, {1,0}, {0,1}}}, renumberedUv{{{1,0}, {0,1}, {0,0}}};
        const omm::CpuInputView original{indices, uv}, equivalent{renumbered, renumberedUv}, differentOrder{reordered, uv};
        Require(original.Hash().has_value() && equivalent.Hash().has_value() && differentOrder.Hash().has_value(),
            "valid triangle inputs must hash");
        Require(original == equivalent && original.Hash() == equivalent.Hash(),
            "vertex renumbering must preserve ordered UV equality");
        Require(!(original == differentOrder), "triangle-corner order must matter");

        omm::CpuInputIdentityRegistry registry;
        // All views passed Hash validation above. Force a collision to test exact equality.
        const uint64_t id = registry.Intern(original, 7);
        Require(id != 0 && registry.Intern(equivalent, 7) == id, "equal colliding inputs must share");
        Require(registry.Intern(differentOrder, 7) != id, "unequal colliding inputs must not share");

        auto negativeZero = uv;
        negativeZero[0].x = std::bit_cast<float>(uint32_t(0x80000000));
        auto nan1 = uv, nan2 = uv;
        nan1[0].x = std::bit_cast<float>(uint32_t(0x7fc00001));
        nan2[0].x = std::bit_cast<float>(uint32_t(0x7fc00002));
        const omm::CpuInputView signedZero{indices, negativeZero}, firstNan{indices, nan1}, secondNan{indices, nan2};
        Require(signedZero.Hash() && firstNan.Hash() && secondNan.Hash(), "raw float payloads remain valid inputs");
        Require(!(original == signedZero) && registry.Intern(signedZero, 7) != id, "signed zero bits must remain distinct");
        Require(!(firstNan == secondNan) && registry.Intern(firstNan, 7) != registry.Intern(secondNan, 7),
            "NaN payload bits must remain distinct");
        Require(firstNan == firstNan, "identical NaN bits must compare equal");

        const std::array<uint32_t, 6> twoTriangles{0, 1, 2, 0, 1, 2};
        const omm::CpuInputView longer{twoTriangles, uv};
        Require(longer.Hash().has_value() && !(original == longer)
            && registry.Intern(longer, 7) != id, "triangle count must participate in equality");

        omm::CpuInputIdentityRegistry nextBatch;
        Require(nextBatch.Intern(original, 7) != id, "new upload batches must not reuse prior IDs");
    }

    void CheckInvalidSources()
    {
        auto mesh = Mesh();
        auto& geometry = *mesh->geometries[0];
        omm::CpuInputIdentityRegistry registry;
        auto reject = [&] { Require(registry.Intern(*mesh, geometry) == 0, "invalid source must not receive an identity"); };
        mesh->buffers->indexData[1] = 3; reject();
        mesh->buffers->indexData[1] = 1;
        mesh->indexOffset = std::numeric_limits<uint32_t>::max(); reject();
        mesh->indexOffset = 0;
        mesh->vertexOffset = std::numeric_limits<uint32_t>::max(); reject();
        mesh->vertexOffset = 0;
        geometry.numIndices = 2; reject();
        geometry.numIndices = 0; reject();
        geometry.numIndices = 3;
        geometry.type = donut::engine::MeshGeometryPrimitiveType::Lines; reject();
        geometry.type = donut::engine::MeshGeometryPrimitiveType::Triangles;
        mesh->buffers->texcoord1Data.clear(); reject();
        mesh->buffers.reset(); reject();

        mesh = Mesh();
        Require(omm::CpuMeshInputIdentity::SupportsStaticMesh(*mesh), "ordinary triangles must be supported");
        mesh->isSkinPrototype = true;
        Require(!omm::CpuMeshInputIdentity::SupportsStaticMesh(*mesh), "skin prototypes must be excluded");
        mesh->isSkinPrototype = false;
        mesh->skinPrototype = Mesh();
        Require(!omm::CpuMeshInputIdentity::SupportsStaticMesh(*mesh), "skinned consumers must be excluded");
        mesh->skinPrototype.reset();
        mesh->isMorphTargetAnimationMesh = true;
        Require(!omm::CpuMeshInputIdentity::SupportsStaticMesh(*mesh), "morphing meshes must be excluded");
        mesh->isMorphTargetAnimationMesh = false;
        mesh->type = donut::engine::MeshType::CurvePolytubes;
        Require(!omm::CpuMeshInputIdentity::SupportsStaticMesh(*mesh), "non-triangle meshes must be excluded");
    }

    void CheckUploadLifetime()
    {
        auto first = Mesh(), second = Mesh();
        second->buffers = first->buffers;
        auto& buffers = *first->buffers;
        buffers.indexData = {99,99,99,0,1,2,2,0,1};
        buffers.texcoord1Data = {{99,99},{99,99},{99,99},{0,0},{1,0},{0,1},{1,0},{0,1},{0,0}};
        first->indexOffset = 2; first->vertexOffset = 1;
        second->indexOffset = 5; second->vertexOffset = 4;
        for (auto* mesh : {first.get(), second.get()})
        {
            mesh->geometries[0]->indexOffsetInMesh = 1;
            mesh->geometries[0]->vertexOffsetInMesh = 2;
        }
        omm::CpuMeshInputIdentity firstIdentity(*first), secondIdentity(*second);
        {
            omm::CpuInputIdentityRegistry registry;
            firstIdentity.geometries[0].id = registry.Intern(*first, *first->geometries[0]);
            secondIdentity.geometries[0].id = registry.Intern(*second, *second->geometries[0]);
            Require(firstIdentity.GetGeometryId(0) != 0
                && firstIdentity.GetGeometryId(0) == secondIdentity.GetGeometryId(0),
                "shared BufferGroup subranges must resolve combined mesh/geometry offsets");
        } // Borrowed registry views must be gone before upload frees their source arrays.
        Require(!firstIdentity.IsValidFor(*first), "pre-upload identities must not validate");
        Upload(buffers);
        Require(buffers.indexData.capacity() == 0 && buffers.texcoord1Data.capacity() == 0,
            "upload fixture must actually release source storage");
        Require(firstIdentity.BindUploadedBuffers(*first) && secondIdentity.BindUploadedBuffers(*second),
            "both shared-buffer identities must survive source storage release");
        Require(firstIdentity.GetGeometryId(0) == secondIdentity.GetGeometryId(0)
            && firstIdentity.GetGeometryId(1) == 0, "compact IDs remain valid after upload");
        omm::CpuInputIdentityRegistry later;
        auto equalLaterMesh = Mesh();
        Require(later.Intern(*equalLaterMesh, *equalLaterMesh->geometries[0]) != firstIdentity.GetGeometryId(0),
            "a later equal import must get a fresh batch ID");
        Require(later.Intern(*first, *first->geometries[0]) == 0, "discarded CPU input must not be read again");

        auto rejectChange = [&](auto& field, auto replacement, const char* message) {
            const auto previous = field;
            field = replacement;
            Require(!firstIdentity.IsValidFor(*first), message);
            field = previous;
        };
        rejectChange(first->indexOffset, first->indexOffset + 1, "mesh index offsets must be checked");
        rejectChange(first->vertexOffset, first->vertexOffset + 1, "mesh vertex offsets must be checked");
        rejectChange(first->totalIndices, first->totalIndices + 1, "mesh counts must be checked");
        auto& geometry = *first->geometries[0];
        rejectChange(geometry.indexOffsetInMesh, uint32_t(0), "geometry index offsets must be checked");
        rejectChange(geometry.vertexOffsetInMesh, uint32_t(0), "geometry vertex offsets must be checked");
        rejectChange(geometry.numIndices, uint32_t(6), "geometry index counts must be checked");
        rejectChange(geometry.numVertices, uint32_t(4), "geometry vertex counts must be checked");
        rejectChange(geometry.type, donut::engine::MeshGeometryPrimitiveType::Lines, "primitive type must be checked");
        rejectChange(first->isMorphTargetAnimationMesh, true, "dynamic changes must invalidate snapshots");
        first->geometries.push_back(std::make_shared<donut::engine::MeshGeometry>());
        Require(!firstIdentity.IsValidFor(*first), "geometry correspondence must be checked");
        first->geometries.pop_back();
        auto& range = buffers.getVertexBufferRange(donut::engine::VertexAttribute::TexCoord1);
        rejectChange(range.byteOffset, range.byteOffset + 8, "UV layout must be checked");
        rejectChange(range.byteSize, range.byteSize - 8, "UV range must be checked");
        rejectChange(first->buffers, std::make_shared<donut::engine::BufferGroup>(buffers), "BufferGroup ownership must match");
        rejectChange(buffers.indexBuffer, nvrhi::BufferHandle::Create(new FakeBuffer(36)), "index buffer handle must match");
        rejectChange(buffers.vertexBuffer, nvrhi::BufferHandle::Create(new FakeBuffer(200)), "UV buffer handle must match");
        auto& indexDesc = static_cast<FakeBuffer*>(buffers.indexBuffer.Get())->desc;
        auto& vertexDesc = static_cast<FakeBuffer*>(buffers.vertexBuffer.Get())->desc;
        rejectChange(indexDesc.byteSize, uint64_t(8), "captured index range must fit its buffer");
        rejectChange(vertexDesc.byteSize, uint64_t(8), "captured UV range must fit its buffer");
        rejectChange(indexDesc.canHaveUAVs, true, "writable index buffers must be excluded");
        rejectChange(vertexDesc.canHaveUAVs, true, "writable UV buffers must be excluded");
        Require(firstIdentity.IsValidFor(*first), "restored original bindings must still validate");
    }

    void CheckHandleLifetime()
    {
        auto mesh = Mesh();
        auto identity = std::make_unique<omm::CpuMeshInputIdentity>(*mesh);
        identity->geometries[0].id = 1;
        Upload(*mesh->buffers);
        Require(identity->BindUploadedBuffers(*mesh), "uploaded immutable buffers must bind");
        mesh->buffers->indexBuffer = nvrhi::BufferHandle::Create(new FakeBuffer(12));
        Require(FakeBuffer::alive == 3 && !identity->IsValidFor(*mesh),
            "snapshot must keep replaced buffer alive, preventing address reuse");
        identity.reset();
        Require(FakeBuffer::alive == 2, "discarding identity must release its old buffer reference");
    }
}

void RunCpuIdentityChecks()
{
    CheckExactEquality();
    CheckInvalidSources();
    CheckUploadLifetime();
    Require(FakeBuffer::alive == 0, "upload checks must release their buffers");
    CheckHandleLifetime();
    Require(FakeBuffer::alive == 0, "identity invalidation must not leak buffer references");
    std::puts("OMM CPU input identity checks passed");
}
