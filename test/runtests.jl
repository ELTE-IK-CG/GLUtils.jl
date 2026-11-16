using GLUtils
using Test
using StaticArrays

@static if VERSION >= v"1.12"
    const FieldAccessError = FieldError
else
    const FieldAccessError = ErrorException
end

const _LIB_MODULE::Module = GLUtils

@testset "Vec types        " begin
    @test VecTN{Float32,2} == VecNT{2,Float32}
    @test sizeof(Vec2T{Bool}) == sizeof(BVec2)
    @test sizeof(Vec3) == 3*sizeof(Float32) # 12
    @test sizeof(DVec2) == 2*sizeof(Float64)
    @test sizeof(Vec3) == sizeof(Vec3(1,2,3))
    @test Vec3 == typeof(vec3(1,2,3))
    @test Vec2(1,2) isa FieldVector
    @test Vec4T{Float32}(1,2,3,4) isa FieldVector
end

@testset "Vec constructors " begin
    @test Vec2(1,2) == vec2(1,2)
    @test Vec3(-3,-3,-3) == Vec3(-3)
    @test zero(IVec4) == ivec4(0)
    @test dvec3(-1,0,1) == vec3(ivec2(-1,0),1)
    @test vec3(0,1,2) == vec3(0,vec2(1,2))
    @test vec4(vec3(1),0) == Vec4(1,vec3(1,1,0))
    @test vec4(vec2(3,-4),vec2(-4,-1)) == vec4(3,-vec2(4),-1)
    @test_throws DimensionMismatch dvec2(1,2,3)
    @test_throws DimensionMismatch uvec4(1,2,3)
    @test_throws DimensionMismatch vec4(vec3(),vec2())
end

@testset "Vec member access" begin
    a = rand(Vec3)
    @test a.x === a[1] && a.y === a[2] && a.z === a[3]
    @test a[(3,)] === a.z
    @test a.yx == a[(2,1)]
    @test a.zx.y == a.x
    @test a.zzzz == vec4(a.z)
    b = rand(IVec4)
    @test b.wzyx.wzyx == b
    @test_throws BoundsError a[4]
    @test_throws BoundsError b[(5,)]
    @test_throws FieldAccessError a.w
    @test_throws FieldAccessError a.u
    @test_throws FieldAccessError b.xyzwx
end

@testset "Mat type creation" begin
    @test MatTN{Float32,2} == MatNT{2,Float32}
    @test DMat3 == DMat3x3
    @test sizeof(IMat4) == 4*sizeof(IVec4) == 4*4*sizeof(Int32)
    @test sizeof(Mat2) == sizeof(Mat2(1,2,3,4))
    @test Mat2(1,2,3,4) isa SMatrix
    @test Mat2T{Int128}(1,2,3,4) isa Mat2T
    @test Mat3(0,0,0,0,0,0,0,0,0) == mat3(0) == zero(Mat3)
    @test Mat2(1,0,0,1) == mat2(1) == one(Mat2)
end

@testset "Others           " begin
    A = mat2(1,2,3,4)
    @test _LIB_MODULE.vec2(A[:,1]) == vec2(1,2)
    eye,at,up = rand(Vec3),rand(Vec3),rand(Vec3)
    @test lookat(eye,at,up)*vec4(eye,1) ≈ vec4(0,0,0,1)
    @test string(A) isa String
    @test string(eye) isa String
end

const _FN_TEST_ATOL = 1e-6

@testset "Trig Fns         " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test _LIB_MODULE.radians(0.0f0) == 0
    @test _LIB_MODULE.radians(180.0f0) ≈ pi
    @test _LIB_MODULE.radians(360.0f0) ≈ 2*pi
    @test _LIB_MODULE.degrees(0.0f0) == 0
    @test _LIB_MODULE.degrees(pi |> Float32) ≈ 180
    @test _LIB_MODULE.degrees(2 * pi |> Float32) ≈ 360

    @test _LIB_MODULE.sin(vec3(0, pi/2, 3*pi/2)) ≈ vec3(0, 1, -1)
    @test _LIB_MODULE.asin(vec3(0, 1, -1)) ≈ vec3(0, pi/2, -pi/2)
    @test _LIB_MODULE.cos(vec3(0, pi/2, pi)) ≈ vec3(1, 0, -1)
    @test _LIB_MODULE.acos(vec3(1, 0, -1)) ≈ vec3(0, pi/2, pi)
    @test _LIB_MODULE.tan(vec3(0, pi/4, pi/6)) ≈ vec3(0, 1, √3/3)
    @test _LIB_MODULE.atan(vec3(0, 1, √3/3)) ≈ vec3(0, pi/4, pi/6)
    @test _LIB_MODULE.atan(vec3(0, 1, √3), vec3(1, 1, 3)) ≈ vec3(0, pi/4, pi/6)

    @test _LIB_MODULE.sinh(vec3(0, log(2), log(3))) ≈ vec3(0, 3/4, 4/3)
    @test _LIB_MODULE.asinh(vec3(0, 3/4, 4/3)) ≈ vec3(0, log(2), log(3))
    @test _LIB_MODULE.cosh(vec3(0, log(2), log(3))) ≈ vec3(1, 5/4, 5/3)
    @test _LIB_MODULE.acosh(vec3(1, 5/4, 5/3)) ≈ vec3(0, log(2), log(3))
    @test _LIB_MODULE.tanh(vec3(0, log(2), log(3))) ≈ vec3(0, 3/5, 4/5)
    @test _LIB_MODULE.atanh(vec3(0, 3/5, 4/5)) ≈ vec3(0, log(2), log(3))
end

@testset "Exponential Fns  " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test _LIB_MODULE.pow(vec4(2), vec4(0,1,2,3)) == vec4(1,2,4,8)
    @test _LIB_MODULE.exp(vec3(0,1,2)) ≈ vec3(1,ℯ,ℯ^2)
    @test _LIB_MODULE.exp2(vec4(0,1,2,3)) ≈ vec4(1,2,4,8)
    @test _LIB_MODULE.log(vec3(1,ℯ,ℯ^2)) ≈ vec3(0,1,2)
    @test _LIB_MODULE.log2(vec4(1,2,4,8)) ≈ vec4(0,1,2,3)
    @test _LIB_MODULE.sqrt(vec4(1,4,9,2)) ≈ vec4(1,2,3,√2)
    @test _LIB_MODULE.inversesqrt(vec4(1,4,9,2)) ≈ vec4(1,0.5,1/3,1/√2)
end

@testset "Common Fns       " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test _LIB_MODULE.abs(vec3(-3,0,2)) == vec3(3,0,2)
    @test _LIB_MODULE.sign(vec3(3,0,-4)) == vec3(1,0,-1)
    @test _LIB_MODULE.floor(vec4(1.1,1,1.9,-1.3)) == vec4(1,1,1,-2)
    @test _LIB_MODULE.trunc(vec4(1.1,1,1.9,-1.3)) == vec4(1,1,1,-1)
    @test _LIB_MODULE.roundEven(vec4(2.4,1.6,1.5,2.5)) == vec4(2)
    @test _LIB_MODULE.round(vec2(2.4, 1.6)) == vec2(2)
    @test _LIB_MODULE.ceil(vec4(0.2,1,-1.2,-1.8)) == vec4(1,1,-1,-1)
    @test _LIB_MODULE.fract(vec3(1.2,2,3.8)) ≈ vec3(0.2,0,0.8)
    @test _LIB_MODULE.mod(vec4(1,2,3,7), vec4(2,2,2,4)) == vec4(1,0,1,3)

    int_out = Ref(vec3(0))
    frac_out = _LIB_MODULE.modf(vec3(1.2, 5, 3.9), int_out)
    @test int_out[] == vec3(1,5,3)
    @test frac_out ≈ vec3(0.2, 0, 0.9)

    @test _LIB_MODULE.min(vec3(1,2,3), vec3(3,2,1)) == vec3(1,2,1)
    @test _LIB_MODULE.min(vec3(1,2,3), 2.0f0) == vec3(1,2,2)
    @test _LIB_MODULE.max(vec3(1,2,3), vec3(3,2,1)) == vec3(3,2,3)
    @test _LIB_MODULE.max(vec3(1,2,3), 2.0f0) == vec3(2,2,3)
    @test _LIB_MODULE.clamp(vec4(3,1,3,4), vec4(1,2,1,5), vec4(5,3,2,5)) == vec4(3,2,2,5)
    @test _LIB_MODULE.clamp(vec3(0,2,5), 1.0f0, 3.0f0) == vec3(1,2,3)
    @test _LIB_MODULE.mix(vec3(0), vec3(2), vec3(0,0.5,1)) == vec3(0,1,2)
    @test _LIB_MODULE.mix(vec3(0), vec3(2), 0.5f0) == vec3(1)
    @test _LIB_MODULE.mix(vec3(0), vec3(2,1,3), bvec3(true, false, true)) == vec3(2,0,3)
    @test _LIB_MODULE.step(vec3(1,0,2), vec3(0,1,2)) == vec3(0,1,1)
    @test _LIB_MODULE.step(1.0f0, vec3(0,1,2)) == vec3(0,1,1)
    @test _LIB_MODULE.smoothstep(vec3(0), vec3(1,2,2), vec3(-1,3,1)) == vec3(0,1,0.5)
    @test _LIB_MODULE.smoothstep(0.0f0, 2.0f0, vec3(-1,3,1)) == vec3(0,1,0.5)
end

@testset "Geometry Fns     " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test _LIB_MODULE.length(vec2(1)) ≈ √2
    @test _LIB_MODULE.length(vec3(1)) ≈ √3
    @test _LIB_MODULE.normalize(vec2(1)) ≈ vec2(1/√2)
    @test _LIB_MODULE.normalize(vec3(1)) ≈ vec3(1/√3)
    @test _LIB_MODULE.distance(vec3(1), vec3(2)) ≈ √3
    @test _LIB_MODULE.distance(vec3(4), vec3(2)) ≈ 2√3
    @test _LIB_MODULE.dot(vec3(1,0,0), vec3(0,1,0)) ≈ 0
    @test _LIB_MODULE.dot(vec3(0.5,2,1), vec3(4,2,3)) ≈ 9
    @test _LIB_MODULE.cross(vec3(1,0,0), vec3(0,1,0)) ≈ vec3(0,0,1)
    @test _LIB_MODULE.cross(vec3(1,2,3), vec3(2,4,6)) ≈ vec3(0,0,0)
    @test _LIB_MODULE.faceforward(vec3(1,0,0), vec3(1,0,0), vec3(1,0,0)) == vec3(-1,0,0)
    @test _LIB_MODULE.faceforward(vec3(1,0,0), vec3(0,-1,0), vec3(0,1,0)) == vec3(1,0,0)
    @test _LIB_MODULE.reflect(vec3(1,-1,0), vec3(0,1,0)) ≈ vec3(1,1,0)
    @test _LIB_MODULE.reflect(vec3(1,1,0), vec3(1,0,1) ./ Float32(√2)) ≈ vec3(0,1,-1)
    @test _LIB_MODULE.refract(vec3(0,-1,0), vec3(0,1,0), 1.0f0) == vec3(0,-1,0)
    @test _LIB_MODULE.refract(vec3(1/√2,-1/√2,0), vec3(0,1,0), 1.0f0) == vec3(1/√2,-1/√2,0)
    @test _LIB_MODULE.refract(vec3(1/√2,-1/√2,0), vec3(0,1,0), 2.0f0) == vec3(0)
end

@testset "Relational Fns   " begin
    @test _LIB_MODULE.lessThan(vec3(0,1,2), vec3(2,1,1)) == bvec3(true,false,false)
    @test _LIB_MODULE.lessThanEqual(vec3(0,1,2), vec3(2,1,1)) == bvec3(true,true,false)
    @test _LIB_MODULE.greaterThan(vec3(0,1,2), vec3(2,1,1)) == bvec3(false,false,true)
    @test _LIB_MODULE.greaterThanEqual(vec3(0,1,2), vec3(2,1,1)) == bvec3(false,true,true)
    @test _LIB_MODULE.equal(vec3(1,1,2), vec3(1,3,1)) == bvec3(true,false,false)
    @test _LIB_MODULE.notEqual(vec3(1,1,2), vec3(1,3,1)) == bvec3(false,true,true)
    @test _LIB_MODULE.any(bvec2(true,false))
    @test !_LIB_MODULE.any(bvec2(false))
    @test _LIB_MODULE.all(bvec2(true))
    @test !_LIB_MODULE.all(bvec2(true,false))
    @test _LIB_MODULE.not(bvec2(true,false)) == bvec2(false,true)
end