open Tensor

type v = { id : int; data : t; local_gradients : (v * (t -> t)) list;  operation : string; }

module VariableHashtbl : sig
  type 'a t
end
(* Hashtable of variables used for the computational graph*)

val make : ?local_gradients:(v * (t -> t)) list -> ?operation:string -> t -> v
(* Create the variable from a Tensor *)

val create : ?dims:dims -> float -> v
(** [create dims value] creates a variable tensor (scalar, vector, or matrix) with the specified 
        dimensions, filled with float [value]. *)

val zero : unit -> v
(* Create a variable with 0 value*)

val one : unit -> v
(* Create a variable with 0 value*)

val zeros : dims -> v
(* Create a variable with random value*)

val random : ?seed:int -> dims -> v
(* Create a variable with random value*) 

val get : v -> dims -> float
(* Gets value from variable *)

val add : v -> v -> v
(* Create the variable that results from adding two variables *)

val mul : v -> v -> v
(* Create the variable that results from multiplying two variables *)

val div : v -> v -> v
(* Create the variable that results from multiplying two variables *)

val neg : v -> v
(* Negate the variable *)

val inv : v -> v
(* Inverse of the variable *)

val sub : v -> v -> v
(* Create the variable that results from subtracting two variables *)

val compare : v -> v -> int
(* Compare two variable float values *)

val equal : v -> v -> bool
(* Check if two variable float values are equal*)

val sin : v -> v
(* Sine of the variable *)

val cos : v -> v
(* Cosine of the variable *)

val tan : v -> v
(* Tangent of the variable *)

val exp : v -> v
(* exp of the variable *)

val log : v -> v
(* log of the variable *)

val gradients : v -> t VariableHashtbl.t
(* Compute the *local gradients* of all the variable *)

val find : t VariableHashtbl.t -> v -> t
(* Find the local gradient of a variable *)

(* Tensor operations with gradient calculations *)

val sum : ?axis:int -> v -> v
(** [sum v] computes the sum of all elements in the tensor, returning a scalar value. *)

(* Machine Learning Operations *)

val matmul : v -> v -> v
(** [matmul v1 v2] performs matrix multiplication on two matrices.
        The number of columns in the first matrix must match the number of rows in the
        second matrix. *)

val transpose : v -> v
(** [transpose v] transposes the matrix. This operation is only applicable for matrices,
                    swapping rows and columns. *)

val softmax : ?axis:int -> v -> v

(* Operator overloading *) 

val ( + ) : v -> v -> v
val ( - ) : v -> v -> v
val ( * ) : v -> v -> v
val ( / ) : v -> v -> v
val ( = ) : v -> v -> bool

(* Printing Ops *)

val print : v -> unit
(* Print the value of a variable *)

val print_table : t VariableHashtbl.t -> unit
(* Prints the gradient hash_table. Useful for looking at overall composition of functions. *)

(* Machine Learning Function *)

val sigmoid : v -> v
val leaky_relu : ?alpha:float -> v -> v
val binary_cross_entropy : v -> v -> v


val visualize : v -> string -> unit
(* Visualize the computation graph of a variable *)
val broadcastinfo : dims -> dims -> dims * dims
val enable_broadcast : ?matmul:bool -> v -> v -> v * v
