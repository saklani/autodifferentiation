open Bigarray
open OUnit2
open Core

module Test = struct
  module T = Tensor
  module V = Variable

  (* Verify all elements fulfil condition *)
  let check_all t f =
    let dims = T.shape t in
    let rec check_recursive indices current_dim =
      if current_dim = Array.length dims then f @@ T.get t indices
      else
        for i = 0 to dims.(current_dim) - 1 do
          let new_indices = Array.append indices [| i |] in
          check_recursive new_indices (current_dim + 1)
        done
    in
    check_recursive [||] 0

  let test_zeros _ =
    (* Test scalar (0-dimensional) tensor *)
    let scalar = T.zeros [||] in
    assert_equal [||] @@ T.shape scalar;
    assert_equal 0.0 @@ T.get scalar [||];

    (* Test 1-dimensional vector *)
    let vector = T.zeros [| 1 |] in
    assert_equal [| 1 |] @@ T.shape vector;
    assert_equal 0.0 @@ T.get vector [| 0 |];

    (* Test 2-dimensional matrices *)
    let matrix_1x2 = T.zeros [| 1; 2 |] in
    assert_equal [| 1; 2 |] @@ T.shape matrix_1x2;
    assert_equal 0.0 @@ T.get matrix_1x2 [| 0; 0 |];
    assert_equal 0.0 @@ T.get matrix_1x2 [| 0; 1 |];

    (* Test 3-dimensional tensor *)
    let tensor_1x2x3 = T.zeros [| 1; 2; 3 |] in
    assert_equal [| 1; 2; 3 |] @@ T.shape tensor_1x2x3;

    let f el = assert_equal 0.0 el in

    (* Check all zeros for different dimensional tensors *)
    check_all vector f;
    check_all matrix_1x2 f;
    check_all tensor_1x2x3 f;

    (* Additional dimension tests *)
    let large_matrix = T.zeros [| 10; 20 |] in
    assert_equal [| 10; 20 |] @@ T.shape large_matrix;
    assert_equal 0.0 @@ T.get large_matrix [| 5; 10 |];

    (* Edge cases *)
    let zero_matrix = T.zeros [| 0; 5 |] in
    assert_equal [| 0; 5 |] @@ T.shape zero_matrix;

    (* Test very large dimensional tensor *)
    let large_tensor = T.zeros [| 1; 2; 3; 4; 5 |] in
    assert_equal [| 1; 2; 3; 4; 5 |] @@ T.shape large_tensor;

    (* Performance and memory allocation test *)
    let huge_matrix = T.zeros [| 10000; 10000 |] in
    assert_equal [| 10000; 10000 |] @@ T.shape huge_matrix;
    assert_equal 0.0 @@ T.get huge_matrix [| 50; 50 |]

  let test_ones _ =
    (* Scalar (0-dimensional) tensor *)
    let scalar = T.ones [||] in
    assert_equal [||] @@ T.shape scalar;
    assert_equal 1.0 @@ T.get scalar [||];

    (* 1-dimensional vector *)
    let vector = T.ones [| 1 |] in
    assert_equal [| 1 |] @@ T.shape vector;
    assert_equal 1.0 @@ T.get vector [| 0 |];

    (* 2-dimensional matrices *)
    let matrix_1x2 = T.ones [| 1; 2 |] in
    assert_equal [| 1; 2 |] @@ T.shape matrix_1x2;
    assert_equal 1.0 @@ T.get matrix_1x2 [| 0; 0 |];
    assert_equal 1.0 @@ T.get matrix_1x2 [| 0; 1 |];

    (* 3-dimensional tensor *)
    let tensor_1x2x3 = T.ones [| 1; 2; 3 |] in
    assert_equal [| 1; 2; 3 |] @@ T.shape tensor_1x2x3;

    (* Verify all elements are one for various dimensions *)
    let f el = assert_equal 1.0 el in
    check_all vector f;
    check_all matrix_1x2 f;
    check_all tensor_1x2x3 f;

    (* Edge cases *)
    let large_matrix = T.ones [| 10; 20 |] in
    assert_equal [| 10; 20 |] @@ T.shape large_matrix;
    assert_equal 1.0 @@ T.get large_matrix [| 5; 10 |];

    (* Zero-length matrix *)
    let zero_matrix = T.ones [| 0; 5 |] in
    assert_equal [| 0; 5 |] @@ T.shape zero_matrix;

    (* High-dimensional tensor *)
    let high_dim_tensor = T.ones [| 1; 2; 3; 4; 5 |] in
    assert_equal [| 1; 2; 3; 4; 5 |] @@ T.shape high_dim_tensor

  let test_random _ =
    (* Scalar (0-dimensional) tensor *)
    let scalar = T.random [||] in
    assert_equal [||] @@ T.shape scalar;
    assert_bool "Scalar should be between 0 and 1"
      (let t = T.get scalar [||] in
       Float.compare t 0.0 <= 1 && Float.compare t 1.0 <= -1);

    (* Test seeded randomness *)
    let with_seed = T.random ~seed:42 [| 5 |] in
    let seeded_again = T.random ~seed:42 [| 5 |] in

    (* Verify that seeded random generates same tensor *)
    assert_equal seeded_again with_seed;

    (* Check tensor shapes *)
    assert_equal [| 1 |] @@ T.shape (T.random [| 1 |]);
    assert_equal [| 1; 2 |] @@ T.shape (T.random [| 1; 2 |]);
    assert_equal [| 1; 2; 3 |] @@ T.shape (T.random [| 1; 2; 3 |]);

    (* Verify random values are in [0, 1) range *)
    let f t =
      assert_bool "Random value out of range"
        (Float.compare t 0.0 <= 1 && Float.compare t 1.0 <= -1)
    in
    check_all (T.random [| 10; 20 |]) f;
    check_all (T.random [| 5; 5; 5 |]) f

  let test_map _ =
    (* Constant mapping *)
    let test_constant_map dims constant =
      let random = T.random dims in
      let mapped = T.map (fun _ -> constant) random in
      let expected =
        if Float.compare constant 0.0 = 0 then T.zeros dims
        else if Float.compare constant 1.0 = 0 then T.ones dims
        else T.random dims |> T.map (fun _ -> constant)
      in
      assert_equal expected mapped
    in

    (* Test mapping to zero *)
    test_constant_map [||] 0.0;
    test_constant_map [| 1 |] 0.0;
    test_constant_map [| 1; 2; 3 |] 0.0;

    (* Test mapping to one *)
    test_constant_map [||] 1.0;
    test_constant_map [| 1 |] 1.0;
    test_constant_map [| 1; 2; 3 |] 1.0;

    (* Test custom mapping *)
    let square_map tensor = T.map (fun x -> x *. x) tensor in
    let t = Genarray.init float32 c_layout [| 2; 2 |] (fun _ -> 3.0) in
    let squared = square_map t in
    (* Verify each element is squared *)
    let dims = T.shape t in
    let rec check_squared indices current_dim =
      if current_dim = Array.length dims then
        let original = T.get t indices in
        let sq = T.get squared indices in
        assert_equal (original *. original) sq
      else
        for i = 0 to dims.(current_dim) - 1 do
          let new_indices = Array.append indices [| i |] in
          check_squared new_indices (current_dim + 1)
        done
    in
    check_squared [||] 0

  let test_map2 _ =
    (* Comprehensive map2 tests *)
    let test_map2_op dims op expected_op =
      let zero_tensor = T.zeros dims in
      let one_tensor = T.ones dims in

      (* Test with zeros and ones *)
      let zero_one_result = T.map2 op zero_tensor one_tensor in
      let one_zero_result = T.map2 op one_tensor zero_tensor in

      (* Verify element-wise operations *)
      let dims = T.shape zero_tensor in
      let rec check_map2 indices current_dim =
        if current_dim = Array.length dims then (
          let zero_one_val = T.get zero_one_result indices in
          let one_zero_val = T.get one_zero_result indices in
          let expected_zero_one = expected_op 0.0 1.0 in
          let expected_one_zero = expected_op 1.0 0.0 in
          assert_equal expected_zero_one zero_one_val;
          assert_equal expected_one_zero one_zero_val)
        else
          for i = 0 to dims.(current_dim) - 1 do
            let new_indices = Array.append indices [| i |] in
            check_map2 new_indices (current_dim + 1)
          done
      in
      check_map2 [||] 0
    in

    (* Test various operations *)
    test_map2_op [| 1; 2; 3 |] ( +. ) ( +. );
    test_map2_op [| 1; 2; 3 |] ( -. ) ( -. );
    test_map2_op [| 1; 2; 3 |] ( *. ) ( *. );

    (* Dimension mismatch test *)
    assert_raises (T.DimensionMismatch "(1, 2) and (2, 1)") (fun _ ->
        T.map2 ( +. ) (T.ones [| 1; 2 |]) (T.ones [| 2; 1 |]))

  let test_sum _ =
    (* Scalar sum *)
    let scalar = T.ones [||] in
    assert_equal scalar @@ T.sum scalar;

    (* Full tensor sum for different dimensions *)
    assert_equal 1.0 @@ T.get (T.sum (T.ones [| 1 |])) [||];
    assert_equal 2.0 @@ T.get (T.sum (T.ones [| 1; 2 |])) [||];
    assert_equal 6.0 @@ T.get (T.sum (T.ones [| 1; 2; 3 |])) [||];

    (* Random tensor full sum *)
    let test_random_full_sum dims =
      let random = T.random dims in
      let sum_val = T.get (T.sum random) [||] in
      assert_bool "Sum should be non-negative" (Float.compare sum_val 0.0 >= 0);
      assert_bool "Sum should be less than tensor size"
        (Float.compare sum_val
           (float_of_int (List.fold ~f:( * ) ~init:1 (Array.to_list dims)))
        = -1)
    in

    test_random_full_sum [| 10 |];
    test_random_full_sum [| 5; 5 |];
    test_random_full_sum [| 2; 3; 4 |];

    (* Axis-wise summation tests *)
    let matrix =
      Genarray.init float32 c_layout [| 3; 4 |] (fun i ->
          float_of_int ((i.(0) * 4) + i.(1) + 1))
    in

    (* Sum along axis 0 (column-wise) *)
    let column_sum = T.sum ~axis:0 matrix in

    assert_equal [| 4 |] @@ T.shape column_sum;
    assert_equal ~printer:string_of_float 15.0 @@ T.get column_sum [| 0 |];
    assert_equal ~printer:string_of_float 18.0 @@ T.get column_sum [| 1 |];
    assert_equal ~printer:string_of_float 21.0 @@ T.get column_sum [| 2 |];
    assert_equal ~printer:string_of_float 24.0 @@ T.get column_sum [| 3 |];

    (* Sum along axis 1 (row-wise) *)
    let row_sum = T.sum ~axis:1 matrix in
    assert_equal [| 3 |] @@ T.shape row_sum;
    assert_equal ~printer:string_of_float 10.0 @@ T.get row_sum [| 0 |];
    assert_equal 10.0 @@ T.get row_sum [| 0 |];
    assert_equal 26.0 @@ T.get row_sum [| 1 |];
    assert_equal 42.0 @@ T.get row_sum [| 2 |];

    (* Sum of last row *)

    (* 3D tensor axis summation *)
    let tensor_3d =
      Genarray.init float32 c_layout [| 2; 3; 4 |] (fun i ->
          float_of_int ((i.(0) * 12) + (i.(1) * 4) + i.(2) + 1))
    in

    (* Sum along first axis *)
    let axis0_sum = T.sum ~axis:0 tensor_3d in
    assert_equal [| 3; 4 |] @@ T.shape axis0_sum;

    (* Sum along second axis *)
    let axis1_sum = T.sum ~axis:1 tensor_3d in
    assert_equal [| 2; 4 |] @@ T.shape axis1_sum;

    (* Sum along last axis *)
    let axis2_sum = T.sum ~axis:2 tensor_3d in
    assert_equal [| 2; 3 |] @@ T.shape axis2_sum;

    (* Negative axis indexing *)
    let last_axis_sum = T.sum ~axis:(-1) tensor_3d in
    assert_equal [| 2; 3 |] @@ T.shape last_axis_sum;

    (* Error case: out of bounds axis *)
    assert_raises Tensor.AxisOutOfBounds (fun () -> T.sum ~axis:3 tensor_3d)

  let test_dot _ =
    (* Basic dot product tests *)
    let test_dot_basic dims =
      let ones_vec = T.ones dims in
      let result = T.dot ones_vec ones_vec in
      assert_equal (float_of_int @@ dims.(0)) @@ Genarray.get result [||]
    in

    test_dot_basic [| 1 |];
    test_dot_basic [| 5 |];
    test_dot_basic [| 10 |];

    (* Dot product with specific values *)
    let create_vec values =
      let vec = T.zeros [| Array.length values |] in
      Array.iteri ~f:(fun i v -> Genarray.set vec [| i |] v) values;
      vec
    in

    let vec1 = create_vec [| 1.0; 2.0; 3.0 |] in
    let vec2 = create_vec [| 4.0; 5.0; 6.0 |] in
    let dot_result = T.dot vec1 vec2 in
    assert_equal 32.0 @@ Genarray.get dot_result [||];

    (* Error cases *)
    assert_raises T.OnlyVectorDotProductSupported (fun _ ->
        T.dot (T.ones [| 4 |]) (T.ones [| 5; 4 |]));
    assert_raises (T.DimensionMismatch "(4) and (5)") (fun _ ->
        T.dot (T.ones [| 4 |]) (T.ones [| 5 |]))

  let test_matmul _ =
    (* 1x1 matrix multiplication with ones *)
    assert_equal (T.ones [| 1; 1 |])
    @@ T.matmul (T.ones [| 1; 1 |]) (T.ones [| 1; 1 |]);

    (* 5x4 * 4x5 matrix multiplication *)
    let result = T.matmul (T.ones [| 5; 4 |]) (T.ones [| 4; 5 |]) in
    assert_equal [| 5; 5 |] @@ T.shape result;

    (* Verify result values for known cases *)
    let m1 = Genarray.init float32 c_layout [| 2; 2 |] (fun _ -> 2.0) in
    let m2 = Genarray.init float32 c_layout [| 2; 2 |] (fun _ -> 3.0) in
    let expected = Genarray.init float32 c_layout [| 2; 2 |] (fun _ -> 12.0) in
    assert_equal expected @@ T.matmul m1 m2;

    (* Specific dimension compatibility checks *)
    assert_equal [| 3; 3 |]
    @@ T.shape
    @@ T.matmul (T.ones [| 3; 2 |]) (T.ones [| 2; 3 |]);
    assert_equal [| 4; 4 |]
    @@ T.shape
    @@ T.matmul (T.ones [| 4; 3 |]) (T.ones [| 3; 4 |]);

    (* Error cases *)
    assert_raises (T.DimensionMismatch "(4, 4) and (3, 5)") (fun _ ->
        T.matmul (T.ones [| 4; 4 |]) (T.ones [| 3; 5 |]));

    (* Non-square matrix multiplication *)
    let m3 =
      Genarray.init float32 c_layout [| 2; 3 |] (fun i ->
          match i with
          | [| 0; 0 |] -> 1.0
          | [| 0; 1 |] -> 2.0
          | [| 0; 2 |] -> 3.0
          | [| 1; 0 |] -> 4.0
          | [| 1; 1 |] -> 5.0
          | [| 1; 2 |] -> 6.0
          | _ -> failwith "Unexpected index")
    in

    let m4 =
      Genarray.init float32 c_layout [| 3; 2 |] (fun i ->
          match i with
          | [| 0; 0 |] -> 7.0
          | [| 0; 1 |] -> 8.0
          | [| 1; 0 |] -> 9.0
          | [| 1; 1 |] -> 10.0
          | [| 2; 0 |] -> 11.0
          | [| 2; 1 |] -> 12.0
          | _ -> failwith "Unexpected index")
    in

    let expected_result =
      Genarray.init float32 c_layout [| 2; 2 |] (fun i ->
          match i with
          | [| 0; 0 |] -> 58.0 (* 1*7 + 2*9 + 3*11 *)
          | [| 0; 1 |] -> 64.0 (* 1*8 + 2*10 + 3*12 *)
          | [| 1; 0 |] -> 139.0 (* 4*7 + 5*9 + 6*11 *)
          | [| 1; 1 |] -> 154.0 (* 4*8 + 5*10 + 6*12 *)
          | _ -> failwith "Unexpected index")
    in
    assert_equal expected_result @@ T.matmul m3 m4

  let test_transpose _ =
    (* Test scalar (0-dimensional) tensor *)
    let scalar = T.ones [||] in
    assert_equal scalar @@ T.transpose scalar;

    (* Test 1D vector *)
    let vector = T.ones [| 5 |] in
    assert_equal vector @@ T.transpose vector;

    (* Test 2D matrix transposition *)
    let matrix =
      Genarray.init float32 c_layout [| 2; 3 |] (fun i ->
          match i with
          | [| 0; 0 |] -> 1.0
          | [| 0; 1 |] -> 2.0
          | [| 0; 2 |] -> 3.0
          | [| 1; 0 |] -> 4.0
          | [| 1; 1 |] -> 5.0
          | [| 1; 2 |] -> 6.0
          | _ -> failwith "Unexpected index")
    in

    let expected_transpose =
      Genarray.init float32 c_layout [| 3; 2 |] (fun i ->
          match i with
          | [| 0; 0 |] -> 1.0
          | [| 0; 1 |] -> 4.0
          | [| 1; 0 |] -> 2.0
          | [| 1; 1 |] -> 5.0
          | [| 2; 0 |] -> 3.0
          | [| 2; 1 |] -> 6.0
          | _ -> failwith "Unexpected index")
    in

    assert_equal expected_transpose @@ T.transpose matrix;

    (* Test higher dimensional tensor transposition *)
    let tensor_3d =
      Genarray.init float32 c_layout [| 2; 3; 4 |] (fun i ->
          float_of_int (((i.(0) + 1) * 100) + ((i.(1) + 1) * 10) + (i.(2) + 1)))
    in

    let transposed_3d = T.transpose tensor_3d in
    assert_equal [| 3; 2; 4 |] @@ T.shape transposed_3d;

    (* Verify a few specific values in the transposed tensor *)
    assert_equal
      (Genarray.get tensor_3d [| 0; 0; 0 |])
      (Genarray.get transposed_3d [| 0; 0; 0 |]);
    assert_equal
      (Genarray.get tensor_3d [| 1; 2; 3 |])
      (Genarray.get transposed_3d [| 2; 1; 3 |])

  let test_flatten _ =
    (* Test scalar (0-dimensional) tensor *)
    let scalar = T.ones [||] in
    let flattened_scalar = T.flatten scalar in
    assert_equal [||] @@ T.shape flattened_scalar;
    assert_equal 1.0 @@ T.get flattened_scalar [||];

    (* Test 1D vector *)
    let vector = T.ones [| 5 |] in
    let flattened_vector = T.flatten vector in
    assert_equal vector @@ flattened_vector;

    (* Test 2D matrix flattening *)
    let matrix =
      Genarray.init float32 c_layout [| 2; 3 |] (fun i ->
          match i with
          | [| 0; 0 |] -> 1.0
          | [| 0; 1 |] -> 2.0
          | [| 0; 2 |] -> 3.0
          | [| 1; 0 |] -> 4.0
          | [| 1; 1 |] -> 5.0
          | [| 1; 2 |] -> 6.0
          | _ -> failwith "Unexpected index")
    in

    let flattened_matrix = T.flatten matrix in
    assert_equal [| 6 |] @@ T.shape flattened_matrix;
    assert_equal 1.0 @@ T.get flattened_matrix [| 0 |];
    assert_equal 6.0 @@ T.get flattened_matrix [| 5 |];

    (* Test 3D tensor flattening *)
    let tensor_3d =
      Genarray.init float32 c_layout [| 2; 3; 4 |] (fun i ->
          float_of_int (((i.(0) + 1) * 100) + ((i.(1) + 1) * 10) + (i.(2) + 1)))
    in

    let flattened_3d = T.flatten tensor_3d in
    assert_equal [| 24 |] @@ T.shape flattened_3d;

    (* Verify first and last values *)
    assert_equal 111.0 @@ T.get flattened_3d [| 0 |];
    assert_equal 234.0 @@ T.get flattened_3d [| 23 |]


  (* let test_broadcasting _ =
    (* Test scalar broadcast *)
    let scalar = T.create 2.0 in
    let vector = T.ones [| 3 |] in
    let result = T.add scalar vector in
    assert_equal [| 3 |] @@ T.shape result;
    Array.iter ~f:(fun i -> assert_equal 3.0 @@ T.get result [| i |]) [| 0; 1; 2 |];
  
    (* Test matrix broadcast with compatible dimensions *)
    let mat1 = T.create ~dims:[| 2; 2 |] 2.0 in
    let mat2 = T.ones [| 2; 2 |] in
    let result = T.add mat1 mat2 in
    assert_equal [| 2; 2 |] @@ T.shape result;
    
    (* Test broadcasting failures *)
    assert_raises (T.DimensionMismatch "(2, 3) and (3, 2)") (fun () ->
      let a = T.create ~dims:[| 2; 3 |] 1.0 in
      let b = T.create ~dims:[| 3; 2 |] 1.0 in 
      T.add a b) *)

  let test_where _ =
    let condition = T.create ~dims:[| 2; 2 |] 1.0 in
    let x = T.create ~dims:[| 2; 2 |] 5.0 in
    let y = T.create ~dims:[| 2; 2 |] 3.0 in
    let result = T.where condition x y in
    assert_equal [| 2; 2 |] @@ T.shape result;
    assert_equal 5.0 @@ T.get result [| 0; 0 |]

  let test_swapaxes _ =
    let t = T.create ~dims:[| 2; 3; 4 |] 1.0 in
    let result = T.swapaxes t 0 1 in
    assert_equal [| 3; 2; 4 |] @@ T.shape result



  let series =
    "Given tests"
    >::: [
           "1 - zeros" >:: test_zeros;
           "2 - ones" >:: test_ones;
           "3 - random" >:: test_random;
           "4 - map" >:: test_map;
           "5 - map2" >:: test_map2;
           "6 - sum" >:: test_sum;
           "7 - dot" >:: test_dot;
           "8 - matmul" >:: test_matmul;
           "9 - transpose" >:: test_transpose;
           "10 - flatten" >:: test_flatten;
           "11 - where" >:: test_where;
           "12 - swapaxes" >:: test_swapaxes;
         ]
end

let series = "Tensor tests" >::: [ Test.series ]
let () = run_test_tt_main series
