open Bigarray
open Core
module F = Float

type t = (float, float32_elt, c_layout) Genarray.t
type dims = int array

exception DimensionMismatch of string
exception OnlyVectorDotProductSupported
exception OnlyMatrixProductSupported
exception AxisOutOfBounds

let get = Genarray.get
let shape = Genarray.dims

let create ?(dims : int array = [||]) (value : float) =
  let a = Genarray.create float32 c_layout dims in
  Genarray.fill a value;
  a

let zeros (dims : int array) : t =
  let a = Genarray.create float32 c_layout dims in
  Genarray.fill a 0.0;
  a

let ones (dims : int array) : t =
  let a = Genarray.create float32 c_layout dims in
  Genarray.fill a 1.0;
  a

let random ?seed (dims : int array) : t =
  (match seed with Some s -> Random.init s | None -> ());
  let a = Genarray.init float32 c_layout dims (fun _ -> Random.float 1.0) in
  a

let broadcast (x : t) (y : t) : t * t =
  let d1 = shape x and d2 = shape y in
  let m = Array.length d1 and n = Array.length d2 in
  let a, b =
    match (m, n) with
    | 0, _ ->
        let v = get x [||] in
        (create ~dims:d2 v, y)
    | _, 0 ->
        let v = get y [||] in
        (x, create ~dims:d1 v)
    | _, _ -> (x, y)
  in
  (a, b)

let map (f : float -> float) (t : t) =
  let map_f i = f (Genarray.get t i) in
  let dims = shape t in
  Genarray.init float32 c_layout dims map_f

let map2 f t1 t2 =
  let a, b = broadcast t1 t2 in
  let d1 = shape a and d2 = shape b in
  if Poly.compare d1 d2 <> 0 then
    raise
      (DimensionMismatch
         (Printf.sprintf "(%s) and (%s)"
            (String.concat_array ~sep:", "
            @@ Array.map d1 ~f:(fun e -> string_of_int e))
            (String.concat_array ~sep:", "
            @@ Array.map d2 ~f:(fun e -> string_of_int e))))
  else
    let map_f i = f (Genarray.get a i) (Genarray.get b i) in
    Genarray.init float32 c_layout d1 map_f

(* Element-wise addition *)
let add t1 t2 = map2 ( +. ) t1 t2

(* Element-wise subtraction *)
let sub t1 t2 = map2 ( -. ) t1 t2

(* Element-wise multiplication *)
let mul t1 t2 = map2 ( *. ) t1 t2

(* Element-wise division *)
let div t1 t2 = map2 ( /. ) t1 t2

(* Common operations *)
let log t = map F.log t
let exp t = map F.exp t
let sin t = map F.sin t
let cos t = map F.cos t
let tan t = map F.tan t
let pow t scalar = map (fun e -> e **. scalar) t

(* Negate *)
let neg t = map (fun e -> e *. -1.) t

let incr i dims =
  let l = Array.length i in
  let rec incr_jth k dims j n =
    if k.(j) + 1 < dims.(j) then (
      k.(j) <- k.(j) + 1;
      true)
    else (
      k.(j) <- 0;
      if j + 1 < n then incr_jth k dims (j + 1) n else false)
  in
  incr_jth i dims 0 l

let iter f t =
  let dims = Genarray.dims t in
  let n = Array.length dims in
  let index = Array.init n ~f:(fun _ -> 0) in
  let rec iter_ith f a i =
    f (Genarray.get a i);
    if incr i dims then iter_ith f a i
  in
  iter_ith f t index

let full_sum (t : t) : t =
  let dims = shape t in
  if Array.length dims = 0 then t
  else
    let total : float ref = ref 0.0 in
    iter (fun e -> total := !total +. e) t;
    create !total

let axis_sum (t : t) (ax : int) : t =
  let dims = shape t in
  let n = Array.length dims in

  let axis = if ax < 0 then n + ax else ax in

  (* Validate axis *)
  if axis < 0 || axis >= n then raise AxisOutOfBounds;

  (* Calculate result dimensions *)
  let result_dims = Array.filteri ~f:(fun i _ -> i <> axis) dims in

  (* Create result tensor *)
  let result = zeros result_dims in

  (* Summation logic *)
  let index = Array.init n ~f:(fun _ -> 0) in

  let rec sum_recursive () =
    (* Extract indices for result tensor *)
    let result_index = Array.filteri ~f:(fun i _ -> i <> axis) index in
    (* Get current value *)
    let current_val = Genarray.get t index in
    (* Add to result *)
    let current_result_val =
      try Genarray.get result result_index with _ -> 0.0
    in
    Genarray.set result result_index (current_result_val +. current_val);

    if incr index dims then sum_recursive ()
  in

  sum_recursive ();
  result

(* Sum function with axis reduction support *)
let sum ?axis (t : t) : t =
  match axis with None -> full_sum t | Some ax -> axis_sum t ax

let dot t1 t2 =
  let d1 = shape t1 and d2 = shape t2 in
  if Array.length d1 <> 1 || Array.length d2 <> 1 then
    raise OnlyVectorDotProductSupported
  else if d1.(0) <> d2.(0) then
    raise (DimensionMismatch (Printf.sprintf "(%d) and (%d)" d1.(0) d2.(0)))
  else sum @@ mul t1 t2

(* Matrix product *)
let matmul t1 t2 =
  let d1 = shape t1 and d2 = shape t2 in
  (* Handle scalar cases *)
  if Array.length d1 = 0 then t2
  else if Array.length d2 = 0 then t1 (* 1D x 1D = scalar dot product *)
  else if Array.length d1 = 1 && Array.length d2 = 1 then (
    let result = ref 0.0 in
    for i = 0 to d1.(0) - 1 do
      result := !result +. (Genarray.get t1 [| i |] *. Genarray.get t2 [| i |])
    done;
    create ~dims:[||] !result (* 1D x 2D = matrix-vector multiplication *))
  else if Array.length d1 = 1 && Array.length d2 = 2 then (
    let rows = d2.(0) and cols = d2.(1) in
    if d1.(0) <> cols then
      raise
        (DimensionMismatch
           (Printf.sprintf "Vector (%d) and Matrix (%d, %d)" d1.(0) rows cols))
    else
      let result = zeros [| rows |] in
      for i = 0 to rows - 1 do
        let sum = ref 0.0 in
        for j = 0 to cols - 1 do
          sum := !sum +. (Genarray.get t1 [| j |] *. Genarray.get t2 [| i; j |])
        done;
        Genarray.set result [| i |] !sum
      done;
      result (* 2D x 1D = matrix-vector multiplication *))
  else if Array.length d1 = 2 && Array.length d2 = 1 then (
    let rows = d1.(0) and cols = d1.(1) in
    if d2.(0) <> cols then
      raise
        (DimensionMismatch
           (Printf.sprintf "Matrix (%d, %d) and Vector (%d)" rows cols d2.(0)))
    else
      let result = zeros [| rows |] in
      for i = 0 to rows - 1 do
        let sum = ref 0.0 in
        for j = 0 to cols - 1 do
          sum := !sum +. (Genarray.get t1 [| i; j |] *. Genarray.get t2 [| j |])
        done;
        Genarray.set result [| i |] !sum
      done;
      result (* 2D x 2D = matrix multiplication *))
  else if Array.length d1 = 2 && Array.length d2 = 2 then (
    let r1 = d1.(0) and c1 = d1.(1) and r2 = d2.(0) and c2 = d2.(1) in
    if c1 <> r2 then
      raise
        (DimensionMismatch (Printf.sprintf "(%d, %d) and (%d, %d)" r1 c1 r2 c2))
    else
      let result = zeros [| r1; c2 |] in
      for i = 0 to r1 - 1 do
        for j = 0 to c2 - 1 do
          let sum = ref 0.0 in
          for k = 0 to c1 - 1 do
            sum :=
              !sum +. (Genarray.get t1 [| i; k |] *. Genarray.get t2 [| k; j |])
          done;
          Genarray.set result [| i; j |] !sum
        done
      done;
      result (* Higher dimensional tensors *))
  else
    let n1 = Array.length d1 in
    let n2 = Array.length d2 in

    (* Check if last dimension of t1 matches second-to-last of t2 *)
    if d1.(n1 - 1) <> d2.(n2 - 2) then
      raise
        (DimensionMismatch
           (Printf.sprintf " %s and %s"
              (String.concat ~sep:"x" @@ Array.to_list
              @@ Array.map ~f:string_of_int d1)
              (String.concat ~sep:"x" @@ Array.to_list
              @@ Array.map ~f:string_of_int d2)))
    else
      (* Compute result dimensions *)
      let result_dims =
        Array.init
          (n1 + n2 - 2)
          ~f:(fun i ->
            if i < n1 - 2 then d1.(i)
            else if i = n1 - 2 then d1.(n1 - 2)
            else d2.(n2 - 1))
      in

      let result = zeros result_dims in

      (* Recursive computation for higher dimensions *)
      let index1 = Array.init n1 ~f:(fun _ -> 0) in
      let index2 = Array.init n2 ~f:(fun _ -> 0) in
      let result_index =
        Array.init (Array.length result_dims) ~f:(fun _ -> 0)
      in

      let rec compute_matmul () =
        let sum = ref 0.0 in

        (* Inner product computation *)
        for k = 0 to d1.(n1 - 1) - 1 do
          (* Update indices for inner multiplication *)
          index1.(n1 - 1) <- k;
          index2.(n2 - 2) <- k;

          sum := !sum +. (Genarray.get t1 index1 *. Genarray.get t2 index2)
        done;

        (* Set result *)
        Genarray.set result result_index !sum;

        (* Increment indices *)
        let rec increment_indices dim =
          if dim < 0 then false
          else if dim >= Array.length result_dims then
            increment_indices (dim - 1)
          else (
            result_index.(dim) <- result_index.(dim) + 1;

            if result_index.(dim) >= result_dims.(dim) then (
              result_index.(dim) <- 0;
              increment_indices (dim - 1))
            else true)
        in

        if increment_indices (Array.length result_dims - 1) then
          compute_matmul ()
      in

      compute_matmul ();
      result

(* Transpose operation *)
let transpose t =
  let dims = shape t in
  match Array.length dims with
  | 0 -> t (* Scalar remains unchanged *)
  | 1 -> t (* 1D vector remains unchanged *)
  | 2 ->
      (* For 2D matrix, swap rows and columns *)
      let rows, cols = (dims.(0), dims.(1)) in
      let result = zeros [| cols; rows |] in
      for i = 0 to rows - 1 do
        for j = 0 to cols - 1 do
          Genarray.set result [| j; i |] (Genarray.get t [| i; j |])
        done
      done;
      result
  | n ->
      (* For higher dimensional tensors, swap first two dimensions *)
      let new_dims = Array.copy dims in
      new_dims.(0) <- dims.(1);
      new_dims.(1) <- dims.(0);
      let result = zeros new_dims in

      (* Create index mapping function *)
      let rec transpose_indices current_indices current_depth =
        if current_depth = n then (
          let transposed_indices = Array.copy current_indices in
          transposed_indices.(0) <- current_indices.(1);
          transposed_indices.(1) <- current_indices.(0);
          result_indices current_indices transposed_indices)
        else if current_depth < 2 then
          for i = 0 to dims.(current_depth) - 1 do
            current_indices.(current_depth) <- i;
            transpose_indices current_indices (current_depth + 1)
          done
        else
          for i = 0 to dims.(current_depth) - 1 do
            current_indices.(current_depth) <- i;
            transpose_indices current_indices (current_depth + 1)
          done
      and result_indices orig_indices transposed_indices =
        Genarray.set result transposed_indices (Genarray.get t orig_indices)
      in

      let initial_indices = Array.init n ~f:(fun _ -> 0) in
      transpose_indices initial_indices 0;
      result

(* Flatten operation *)
let flatten t =
  let dims = Genarray.dims t in
  let n = Array.length dims in
  match n with
  | 0 -> t (* Scalar remains unchanged *)
  | 1 -> t (* 1D vector remains unchanged *)
  | _ ->
      let total = Array.fold ~f:( * ) ~init:1 dims in
      let a = Genarray.create float32 c_layout [| total |] in
      let index = Array.init n ~f:(fun _ -> 0) in
      let c = ref 0 in
      let f el =
        Genarray.set a [| !c |] el;
        c := !c + 1
      in
      let rec iter_ith f a i =
        f (Genarray.get a i);
        if incr i dims then iter_ith f a i
      in
      iter_ith f t index;
      a

(* Print *)
let print (t : t) = iter (fun e -> Printf.printf "%f, " e) t
let formatted_print (t : t) =
  let dims = Genarray.dims t in
  let n = Array.length dims in
  let rec print_indices current_indices current_depth =
    if current_depth = n then
      (* Print the value at the current index, followed by a space *)
      Printf.printf "%f " (Genarray.get t current_indices)
    else
      for i = 0 to dims.(current_depth) - 1 do
        current_indices.(current_depth) <- i;
        print_indices current_indices (current_depth + 1)
      done
  in
  let initial_indices = Array.init n ~f:(fun _ -> 0) in
  (* Iterate over the first dimension (rows) *)
  for i = 0 to dims.(0) - 1 do
    initial_indices.(0) <- i;
    Printf.printf "%s" (String.make 2 ' ');
    print_indices initial_indices 1;
    Printf.printf "\n"
  done

(* Reshape *)
let reshape (t : t) (new_dims : dims) : t =
  let old_dims = shape t in
  let old_size = Array.fold ~f:( * ) ~init:1 old_dims in
  let new_size = Array.fold ~f:( * ) ~init:1 new_dims in

  if Int.(old_size <> new_size) then
    raise
      (DimensionMismatch
         (Printf.sprintf "Cannot reshape array of size %d into shape %s"
            old_size
            (String.concat_array ~sep:", "
            @@ Array.map new_dims ~f:string_of_int)))
  else
    let result = zeros new_dims in

    let convert_index idx dims =
      let n = Array.length dims in
      Array.init n ~f:(fun i ->
          let stride =
            Array.fold ~f:( * ) ~init:1
              (Array.sub dims ~pos:(i + 1) ~len:(n - i - 1))
          in
          idx / stride mod dims.(i))
    in

    let flat_idx = ref 0 in
    let rec fill_tensor () =
      if !flat_idx >= old_size then ()
      else
        let old_coords = convert_index !flat_idx old_dims in
        let new_coords = convert_index !flat_idx new_dims in
        Genarray.set result new_coords (Genarray.get t old_coords);
        flat_idx := !flat_idx + 1;
        fill_tensor ()
    in

    fill_tensor ();
    result

(* Swapaxes *)
let swapaxes (t : t) (axis1 : int) (axis2 : int) : t =
  let dims = shape t in
  let n = Array.length dims in

  (* Validate axes *)
  if axis1 < 0 || axis1 >= n || axis2 < 0 || axis2 >= n then
    raise AxisOutOfBounds;

  (* If axes are the same, return original tensor *)
  if axis1 = axis2 then t
  else
    (* Create new dimensions with swapped axes *)
    let new_dims = Array.copy dims in
    new_dims.(axis1) <- dims.(axis2);
    new_dims.(axis2) <- dims.(axis1);

    let result = zeros new_dims in
    let index = Array.create ~len:n 0 in

    let rec fill_tensor () =
      (* Copy current value *)
      let swapped_index = Array.copy index in
      swapped_index.(axis1) <- index.(axis2);
      swapped_index.(axis2) <- index.(axis1);

      Genarray.set result swapped_index (Genarray.get t index);

      (* Move to next position *)
      if incr index dims then fill_tensor ()
    in

    fill_tensor ();
    result

let where (condition : t) (x : t) (y : t) : t =
  let cond_dims = shape condition in

  (* Broadcast the inputs if necessary *)
  let a, b = broadcast x y in
  let broadcast_dims = shape a in

  (* Verify condition dimensions match broadcasted dimensions *)
  if Poly.compare cond_dims broadcast_dims <> 0 then
    raise
      (DimensionMismatch
         (Printf.sprintf "Condition shape %s doesn't match input shapes %s"
            (String.concat_array ~sep:", "
            @@ Array.map cond_dims ~f:string_of_int)
            (String.concat_array ~sep:", "
            @@ Array.map broadcast_dims ~f:string_of_int)));

  let result = zeros broadcast_dims in
  let index = Array.create ~len:(Array.length broadcast_dims) 0 in

  let rec fill_tensor () =
    let cond_val = Genarray.get condition index in
    let value =
      if Float.(Float.abs cond_val > 0.0) then Genarray.get a index
      else Genarray.get b index
    in
    Genarray.set result index value;

    if incr index broadcast_dims then fill_tensor ()
  in

  fill_tensor ();
  result

(* Operator overloading *)
let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( / ) = div
