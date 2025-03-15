(*
   tensor.mli
   This module defines operations on tensors, which can be scalars, vectors, or matrices. *)
open Bigarray

type t = (float, float32_elt, c_layout) Genarray.t
type dims = int array

exception DimensionMismatch of string
exception OnlyVectorDotProductSupported
exception OnlyMatrixProductSupported
exception AxisOutOfBounds

(* Define type for representing shape of tensors (rows and columns) *)

val get : t -> dims -> float
(** [t] is a type representing a tensor, which can either be a Scalar (a single number),
    a Vector (a 1D array of numbers), or a Matrix (a 2D array of numbers).
    [s] represents the shape of a tensor, number of rows and columns. *)

val shape : t -> dims
(** [shape tensor] returns the dimensions of the tensor, 
    record with [rows] and [cols] fields. *)

val create : ?dims:dims -> float -> t
(** [create dims value] creates a tensor (scalar, vector, or matrix) with the specified 
        dimensions, filled with float [value]. *)

val zeros : dims -> t
(** [zeros dims] creates a tensor (scalar, vector, or matrix) with the specified 
    dimensions, filled with zeros. 

    {examples}
    - [zeros [3]] creates a vector of length 3 filled with zeros, 
    - [zeros [2; 3]] creates a 2x3 matrix filled with zeros. *)

val ones : dims -> t
(** [ones dims] creates a tensor (scalar, vector, or matrix) with the specified 
    dimensions, filled with ones. 

    {examples}
    - [ones [3]] creates a vector of length 3 filled with ones, 
    - [ones [2; 3]] creates a 2x3 matrix filled with ones. *)

val random : ?seed:int -> dims -> t
(** [random ?seed dims] creates a tensor (scalar, vector, or matrix) with random 
    values, where [dims] specifies the dimensions of the tensor. 
    Optionally, [seed] can be provided to control the randomness. *)

val map : (float -> float) -> t -> t
(** [map f t] applies the function [f] element-wise to each element of the
       tensor [t], and returns a new tensor with the results. *)

val map2 : (float -> float -> float) -> t -> t -> t
(** [map2 f t1 t2] applies the function [f] element-wise to corresponding
       elements of [t1] and [t2], and returns a new tensor with the results.*)

val add : t -> t -> t
(** [add t1 t2] performs element-wise addition of the two tensors.*)

val sub : t -> t -> t
(** [sub t1 t2] performs element-wise subtraction of the second tensor
       from the first.*)

val mul : t -> t -> t
(** [mul t1 t2] performs element-wise multiplication of the two tensors.*)

val div : t -> t -> t
(** [div t1 t2]  performs element-wise divison of the two tensors. *)

val pow : t -> float -> t
(** [pow t exponent] raises each element of the tensor to the power of [exponent]. *)

val log : t -> t
(** [log t] applies the natural logarithm element-wise to each element of the tensor.
       This is only valid for positive values. *)

val exp : t -> t
(** [exp t] applies the exponential function (e^x) element-wise to each element of the tensor. *)

val sin : t -> t
(** [sin t] applies the sine function element-wise to each element of the tensor. *)

val cos : t -> t
(** [cos t] applies the cosine function element-wise to each element of the tensor. *)

val tan : t -> t
(** [tan t] applies the tangent function element-wise to each element of the tensor. *)

val sum : ?axis:int -> t -> t
(** [sum ?axis t] 
 if axis specified, computes the sum of all elements on an axis in the tensor, returning a scalar value. 
 otherwise, computes the sum of all elements in the tensor
 *)

val dot : t -> t -> t
(** [dot t1 t2] computes the dot product of two vectors. *)

val matmul : t -> t -> t
(** [matmul t1 t2] performs matrix multiplication on two matrices.
        The number of columns in the first matrix must match the number of rows in the
        second matrix. *)

val transpose : t -> t
(** [transpose t] transposes the tensor. This operation is only applicable for matrices,
            swapping rows and columns. *)

val flatten : t -> t
(** [flatten t] flattens the tensor into a one-dimensional vector, regardless of its
                original shape. *)

val reshape : t -> int array -> t
(** [reshape t dims] reshapes the tensor to the specified dimensions. (not supported) *)

val neg : t -> t
(** [neg t] negates each element of the tensor, i.e., multiplies each element by -1. *)

val swapaxes : t -> int -> int -> t
(** [swapaxes t axis1 axis2] Interchange two axes of an array.  *)

val where : t -> t -> t -> t
val print : t -> unit
val formatted_print : t -> unit
val broadcast : t -> t -> t * t

(* Operator overloading *)
val ( + ) : t -> t -> t
(** [t1 + t2] performs element-wise addition of the two tensors. *)

val ( - ) : t -> t -> t
(** [t1 - t2] performs element-wise subtraction of the two tensors. *)

val ( * ) : t -> t -> t
(** [t1 * t2] performs element-wise multiplication of the two tensors. *)

val ( / ) : t -> t -> t
(** [t1 / t2] performs element-wise multiplication of the two tensors.*)
