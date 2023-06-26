using Unity.Jobs;
using UnityEngine.Rendering;
using UnityEngine;
using Unity.Collections;
using System.Runtime.InteropServices;
using Unity.Burst;

public class MeshDataDeformer : BaseDeformer
{
    private Vector3 _positionToDeform;
    private Mesh.MeshDataArray _meshDataArray;
    private Mesh.MeshDataArray _meshDataArrayOutput;
    private VertexAttributeDescriptor[] _layout;
    private SubMeshDescriptor _subMeshDescriptor;
    private DeformMeshDataJob _job;
    private JobHandle _jobHandle;
    private bool _scheduled;

    protected override void Awake()
    {
        base.Awake();
        CreateMeshData();
    }

    private void CreateMeshData()
    {
        _meshDataArray = Mesh.AcquireReadOnlyMeshData(Mesh);
        _layout = new[]
        {
           new VertexAttributeDescriptor(VertexAttribute.Position,_meshDataArray[0].GetVertexAttributeFormat(VertexAttribute.Position), 3),
           new VertexAttributeDescriptor(VertexAttribute.Normal,_meshDataArray[0].GetVertexAttributeFormat(VertexAttribute.Normal), 3),
           new VertexAttributeDescriptor(VertexAttribute.Tangent,_meshDataArray[0].GetVertexAttributeFormat(VertexAttribute.Tangent), 4),
           new VertexAttributeDescriptor(VertexAttribute.TexCoord0,_meshDataArray[0].GetVertexAttributeFormat(VertexAttribute.TexCoord0), 2),
           new VertexAttributeDescriptor(VertexAttribute.TexCoord1,_meshDataArray[0].GetVertexAttributeFormat(VertexAttribute.TexCoord1), 2)
       };
        _subMeshDescriptor =
            new SubMeshDescriptor(0, _meshDataArray[0].GetSubMesh(0).indexCount, MeshTopology.Triangles)
            {
                firstVertex = 0,
                vertexCount = _meshDataArray[0].vertexCount
            };
    }

    private void Update()
    {
        ScheduleJob();
    }

    private void LateUpdate()
    {
        CompleteJob();
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VertexData
    {
        public Vector3 Position;
        public Vector3 Normal;
        public Vector4 Tangent;
        public Vector2 Uv;
        public Vector2 Uv1;
    }

    [BurstCompile]
    public struct DeformMeshDataJob : IJobParallelFor
    {
        public Mesh.MeshData OutputMesh;
        [ReadOnly] private NativeArray<VertexData> _vertexData;
        [ReadOnly] private readonly float _speed;
        [ReadOnly] private readonly float _amplitude;
        [ReadOnly] private readonly float _time;

        public DeformMeshDataJob(
            NativeArray<VertexData> vertexData,
            Mesh.MeshData outputMesh,
            float speed,
            float amplitude,
            float time)
        {
            _vertexData = vertexData;
            OutputMesh = outputMesh;
            _speed = speed;
            _amplitude = amplitude;
            _time = time;
        }

        public void Execute(int index)
        {
            NativeArray<VertexData> outputVertexData = OutputMesh.GetVertexData<VertexData>();
            VertexData vertexData = _vertexData[index];
            Vector3 position = vertexData.Position;
            position.y = DeformerUtilities.CalculateDisplacement(position, _time, _speed, _amplitude);

            outputVertexData[index] = new VertexData
            {
                Position = position,
                Normal = vertexData.Normal,
                Tangent = vertexData.Tangent,
                Uv = vertexData.Uv,
                Uv1 = vertexData.Uv1
            };
        }
    }

    private void ScheduleJob()
    {
        if (_scheduled)
        {
            return;
        }

        _scheduled = true;
        _meshDataArrayOutput = Mesh.AllocateWritableMeshData(1);
        Mesh.MeshData outputMesh = _meshDataArrayOutput[0];
        _meshDataArray = Mesh.AcquireReadOnlyMeshData(Mesh);
        Mesh.MeshData meshData = _meshDataArray[0];
        outputMesh.SetIndexBufferParams(meshData.GetSubMesh(0).indexCount, meshData.indexFormat);
        outputMesh.SetVertexBufferParams(meshData.vertexCount, _layout);

        _job = new DeformMeshDataJob(
            meshData.GetVertexData<VertexData>(),
            outputMesh,
            _speed,
            _amplitude,
            Time.time
        );

        _jobHandle = _job.Schedule(meshData.vertexCount, 64);
    }

    private void CompleteJob()
    {
        if (!_scheduled)
        {
            return;
        }

        _jobHandle.Complete();
        UpdateMesh(_job.OutputMesh);
        _scheduled = false;
    }

    private void UpdateMesh(Mesh.MeshData meshData)
    {
        NativeArray<ushort> outputIndexData = meshData.GetIndexData<ushort>();

        _meshDataArray[0].GetIndexData<ushort>().CopyTo(outputIndexData);
        _meshDataArray.Dispose();

        meshData.subMeshCount = 1;

        meshData.SetSubMesh(0,
                            _subMeshDescriptor,
                            MeshUpdateFlags.DontRecalculateBounds |
                            MeshUpdateFlags.DontValidateIndices |
                            MeshUpdateFlags.DontResetBoneBounds |
                            MeshUpdateFlags.DontNotifyMeshUsers);

        Mesh.MarkDynamic();

        Mesh.ApplyAndDisposeWritableMeshData(_meshDataArrayOutput,
                                             Mesh,
                                             MeshUpdateFlags.DontRecalculateBounds |
                                             MeshUpdateFlags.DontValidateIndices |
                                             MeshUpdateFlags.DontResetBoneBounds |
                                             MeshUpdateFlags.DontNotifyMeshUsers);

        Mesh.RecalculateNormals();
    }
}