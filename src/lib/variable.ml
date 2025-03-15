open Core
open Tensor

(* Variable type *)
type v = {
  id : int;
  data : t;
  local_gradients : (v * (t -> t)) list;
  operation : string;
}

(* Hashtable of variables used for our computational graph *)
module VariableHashtbl = Hashtbl.Make (struct
  type t = v

  let hash v = Hashtbl.hash v.id
  let compare v1 v2 = Poly.compare v1.data v2.data
  let t_of_sexp _ = failwith "Sexp conversion not supported"
  let sexp_of_t _ = failwith "Sexp conversion not supported"
end)

let variable_counter : int ref = ref 0

(* Creates the variable from a Tensor *)
let make ?(local_gradients : (v * (t -> t)) list = [])
    ?(operation : string = "self") (data : t) =
  let id = !variable_counter in
  variable_counter := Int.( + ) !variable_counter 1;
  { id; data; local_gradients; operation }

(* Scalar constructors *)
let const f = make (zeros [||] |> map (fun _ -> f))
let zero () = const 0.0
let one () = const 1.0
let create ?dims (value : float) = make (create ?dims value)
let zeros (dims : dims) = make (zeros dims)
let random ?(seed : int option) (dims : dims) = make (random ?seed dims)
let get (v : v) (dims : dims) : float = get v.data dims

let broadcastinfo (shape_a : int array) (shape_b : int array) :
    int array * int array =
  (* Get maximum number of dimensions *)
  let ndim = max (Array.length shape_a) (Array.length shape_b) in

  (* Calculate how many dimensions to add to each shape *)
  let add_ndims_to_a = Int.(ndim - Array.length shape_a) in
  let add_ndims_to_b = Int.(ndim - Array.length shape_b) in

  (* Create padded shapes with leading ones *)
  let a_shape_ = Array.append (Array.create ~len:add_ndims_to_a 1) shape_a in
  let b_shape_ = Array.append (Array.create ~len:add_ndims_to_b 1) shape_b in

  (* Check if shapes can be broadcast *)
  let can_broadcast =
    Array.for_all2_exn a_shape_ b_shape_ ~f:(fun a b -> a = b || a = 1 || b = 1)
  in

  if not can_broadcast then
    failwith
      (Printf.sprintf "could not broadcast shapes %s %s"
         (Array.to_list shape_a |> List.map ~f:string_of_int
        |> String.concat ~sep:" ")
         (Array.to_list shape_b |> List.map ~f:string_of_int
        |> String.concat ~sep:" "));

  (* Calculate repeat dimensions for a *)
  let a_repeatdims =
    Array.mapi a_shape_ ~f:(fun i a ->
        let is_repeated = a = 1 && Array.get b_shape_ i > 1 in
        let is_added = i < add_ndims_to_a in
        is_repeated || is_added)
  in

  (* Calculate repeat dimensions for b *)
  let b_repeatdims =
    Array.mapi b_shape_ ~f:(fun i b ->
        let is_repeated = b = 1 && Array.get a_shape_ i > 1 in
        let is_added = i < add_ndims_to_b in
        is_repeated || is_added)
  in

  (* Convert boolean arrays to indices where True *)
  let bool_array_to_indices arr =
    Array.mapi arr ~f:(fun i b -> if b then Some i else None)
    |> Array.filter_map ~f:Fn.id
  in

  (bool_array_to_indices a_repeatdims, bool_array_to_indices b_repeatdims)

let enable_broadcast ?(matmul = false) (a : v) (b : v) : v * v =
  (* Get shapes excluding last 2 dimensions if matmul is true *)
  let a_shape =
    if matmul then
      Array.sub (shape a.data) ~pos:0 ~len:Int.(Array.length (shape a.data) - 2)
    else shape a.data
  in
  let b_shape =
    if matmul then
      Array.sub (shape b.data) ~pos:0 ~len:Int.(Array.length (shape b.data) - 2)
    else shape b.data
  in

  let a_repeatdims, b_repeatdims = broadcastinfo a_shape b_shape in

  (* Helper function to sum along specified dimensions and reshape *)
  let sum_and_reshape path_value repeat_dims target_shape =
    let summed =
      Array.fold repeat_dims ~init:path_value ~f:(fun acc dim ->
          sum ~axis:dim acc)
    in
    reshape summed target_shape
  in

  (* Create new variables with broadcast-aware gradients *)
  let a' =
    make a.data
      ~local_gradients:
        [
          ( a,
            fun path_value ->
              let summed =
                sum_and_reshape path_value a_repeatdims (shape a.data)
              in
              Tensor.zeros (shape a.data) + summed );
        ]
  in

  let b' =
    make b.data
      ~local_gradients:
        [
          ( b,
            fun path_value ->
              let summed =
                sum_and_reshape path_value b_repeatdims (shape b.data)
              in
              Tensor.zeros (shape b.data) + summed );
        ]
  in

  (a', b')

(* Tensor-aware operations *)
let add (x : v) (y : v) =
  let a, b = enable_broadcast x y in
  let data = a.data + b.data in
  let local_gradients =
    [
      (x, fun path_value -> path_value);
      (* d/dx (x + y) = 1 *)
      (y, fun path_value -> path_value);
      (* d/dy (x + y) = 1 *)
    ]
  in
  let operation = "add" in
  make data ~local_gradients ~operation

let mul (x : v) (y : v) =
  let a, b = enable_broadcast x y in
  let data = x.data * y.data in
  let local_gradients =
    [
      (x, fun path_value -> path_value * b.data);
      (* d/dx (x * y) = y *)
      (y, fun path_value -> path_value * a.data);
      (* d/dy (x * y) = x *)
    ]
  in
  let operation = "mul" in
  make data ~local_gradients ~operation

let div (x : v) (y : v) =
  let a, b = enable_broadcast x y in
  let data = map2 ( /. ) x.data y.data in
  let local_gradients =
    [
      (x, fun path_value -> path_value / b.data);
      (* d/dx (x / y) = 1/y *)
      (y, fun path_value -> path_value * neg a.data / (b.data * b.data))
      (* d/dy (x / y) = -x/(y*y) *);
    ]
  in
  let operation = "mul" in
  make data ~local_gradients ~operation

let inv (x : v) =
  let data = map (fun v -> 1.0 /. v) x.data in
  let local_gradients =
    [ (x, fun path_value -> neg (path_value / (x.data * x.data))) ]
  in
  let operation = "inv" in
  make data ~local_gradients ~operation

let neg (x : v) =
  let dims = shape x.data in
  let data = neg x.data in
  let local_gradients =
    [ (x, fun path_value -> path_value * neg (ones dims)) ]
  in
  make data ~local_gradients

let sub (x : v) (y : v) = add x @@ neg y

(* Logarithmic operation *)
let log (x : v) =
  let data = log x.data in
  let local_gradients = [ (x, fun path_value -> path_value / x.data) ] in
  let operation = "log" in
  make data ~local_gradients ~operation

(* Exponent operation *)
let exp (x : v) =
  let data = exp x.data in
  let local_gradients = [ (x, fun path_value -> path_value * exp x.data) ] in
  let operation = "exp" in
  make data ~local_gradients ~operation

(* Trigonometric operations *)
let sin (x : v) =
  let data = sin x.data in
  let local_gradients =
    [ (x, fun path_value -> path_value * Tensor.cos x.data) ]
  in
  let operation = "sin" in
  make data ~local_gradients ~operation

let cos (x : v) =
  let data = cos x.data in
  let local_gradients =
    [ (x, fun path_value -> path_value * Tensor.neg (Tensor.sin x.data)) ]
  in
  let operation = "cos" in
  make data ~local_gradients ~operation

let tan (x : v) =
  let data = tan x.data in
  let local_gradients =
    [
      ( x,
        fun path_value ->
          path_value
          * map
              (fun e ->
                let cs = Float.cos e in
                1. /. (cs *. cs))
              x.data );
    ]
  in
  let operation = "tan" in
  make data ~local_gradients ~operation

(* Comparison and equality *)
let compare a b = Float.compare (get a [||]) (get b [||])
let equal a b = compare a b = 0

(* Gradient computation *)
let rec compute (grad_tbl : t VariableHashtbl.t) (var : v) (path_value : t) =
  List.iter
    ~f:(fun (child_variable, multipy_by_locg_f) ->
      (* Multiply edges of a path *)
      let gradient_value_of_path = multipy_by_locg_f path_value in
      (* Add the different paths *)
      let prev_grad =
        match Hashtbl.find grad_tbl child_variable with
        | Some p -> p
        | None -> Tensor.zeros [||]
      in
      Hashtbl.set grad_tbl ~key:child_variable
        ~data:(prev_grad + gradient_value_of_path);
      (* Recurse *)
      compute grad_tbl child_variable gradient_value_of_path)
    var.local_gradients

(* Gradient dictionary *)
let gradients (var : v) : t VariableHashtbl.t =
  let grad_tbl : t VariableHashtbl.t = VariableHashtbl.create () in
  compute grad_tbl var (ones [||]);
  grad_tbl

(* Find gradient for a specific variable *)
let find grad_tbl a =
  let f = Hashtbl.find grad_tbl a in
  match f with Some x -> x | None -> failwith "Gradient not found"

let sum ?axis (x : v) =
  let data =
    match axis with Some axis -> sum ~axis x.data | None -> sum x.data
  in
  let local_gradients = [ (x, fun path_value -> path_value + data) ] in
  make data ~local_gradients

let matmul (x : v) (y : v) =
  let a, b = enable_broadcast ~matmul:true x y in
  let m = Array.length @@ shape a.data and n = Array.length @@ shape b.data in
  let data = matmul x.data y.data in
  let local_gradients =
    [
      (a, fun path_value -> path_value * swapaxes a.data Int.(m - 2) Int.(m - 1));
      (b, fun path_value -> swapaxes b.data Int.(n - 2) Int.(n - 1) * path_value);
    ]
  in
  let operation = "matmul" in
  make data ~local_gradients ~operation

let transpose (v : v) : v =
  let data = transpose v.data in
  let local_gradients = [ (v, fun path_value -> transpose path_value) ] in
  let operation = "transpose" in
  make data ~local_gradients ~operation

(* Machine Learning Functions *)
let softmax ?(axis = -1) (v : v) : v =
  let exp_a = exp v in
  let s = sum ~axis v in
  let data = Tensor.exp v.data in
  let local_gradients = [ (v, fun _ -> exp_a.data / s.data) ] in
  let operation = "softmax" in
  make data ~local_gradients ~operation

(* Leaky ReLU activation function *)
let leaky_relu ?(alpha = 0.01) (x : v) : v =
  let data =
    Tensor.map (fun v -> if Float.(v > 0.0) then v else alpha *. v) x.data
  in
  let local_gradients =
    [
      ( x,
        fun path_value ->
          Tensor.map2
            (fun v grad -> if Float.(v > 0.0) then grad else alpha *. grad)
            x.data path_value );
    ]
  in
  let operation = "leaky_relu" in
  make data ~local_gradients ~operation

(* Operator overloading *)
let ( = ) = equal
let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( / ) = div

(* Printing *)
let print v =
  let dims = shape v.data in
  if Int.equal (Array.length dims) 0 then Printf.printf "%f " (get v [||])
  else
    Printf.printf "Tensor of shape %s\n"
      (String.concat ~sep:"x" @@ Array.to_list
      @@ Array.map ~f:string_of_int dims)

(* Printing *)
let print_table (grad_tbl : t VariableHashtbl.t) =
  Hashtbl.iter_keys grad_tbl ~f:(fun e ->
      Printf.printf "%d %f %s\n" e.id (Tensor.get e.data [||]) e.operation)

(* Sigmoid activation function *)
let sigmoid (x : v) : v =
  let dims = shape x.data in
  let one = create ~dims 1.0 in
  one / (one + exp (neg x))

(* Binary cross-entropy loss *)
let binary_cross_entropy (y_true : v) (y_pred : v) : v =
  let dims = shape y_true.data in
  let epsilon = create ~dims 1e-6 in
  let one = create ~dims 1.0 in
  let term1 = y_true * log (y_pred + epsilon) in
  let term2 = (one - y_true) * log (one - y_pred + epsilon) in
  let loss = neg (sum (term1 + term2)) in
  loss

(*Outputting computation graph*)
let visualize (var : v) (output_file : string) =
  let visited = Hashtbl.create (module Int) in
  let buffer = Buffer.create 1024 in
  Buffer.add_string buffer "digraph computation_graph {\n";

  let rec visit_node (node : v) =
    if not (Hashtbl.mem visited node.id) then (
      Hashtbl.add_exn visited ~key:node.id ~data:();
      (* Add node label *)
      Buffer.add_string buffer
        (Printf.sprintf "  node%d [label=\"id: %d\\ndata: %.2f\"];\n" node.id
           node.id
           (Tensor.get node.data [||]));
      (* Add edges *)
      List.iter node.local_gradients ~f:(fun (child, _) ->
          Buffer.add_string buffer
            (Printf.sprintf "  node%d -> node%d;\n" node.id child.id);
          visit_node child))
  in

  visit_node var;
  Buffer.add_string buffer "}\n";
  Out_channel.write_all output_file ~data:(Buffer.contents buffer);
  Printf.printf "Computation graph written to %s\n" output_file
