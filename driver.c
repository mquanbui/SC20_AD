/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#include <assert.h>
#include <time.h>

#include "interpreter.h"
#include "multivector.h"
#include "HYPRE_MatvecFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

hypre_int
main( hypre_int argc,
    char *argv[] )
{
  HYPRE_Int                 arg_index;
  HYPRE_Int                 build_matrix_arg_index;
  HYPRE_Int                 build_precond_type;
  HYPRE_Int                 build_precond_arg_index;
  HYPRE_Int                 build_rhs_type;
  HYPRE_Int                 build_rhs_arg_index;
  HYPRE_Int                 build_block_cf_arg_index;
  HYPRE_Int                 ierr = 0;
  HYPRE_Int                 i;
  HYPRE_Real          final_res_norm;
  void               *object;

  HYPRE_IJMatrix      ij_A = NULL;
  HYPRE_IJMatrix      ij_M = NULL;
  HYPRE_IJVector      ij_b = NULL;
  HYPRE_IJVector      ij_x = NULL;

  HYPRE_ParCSRMatrix  parcsr_A = NULL;
  HYPRE_ParCSRMatrix  parcsr_M = NULL;
  HYPRE_ParVector     b = NULL;
  HYPRE_ParVector     x = NULL;

  HYPRE_Solver        mgr_coarse_solver;
  HYPRE_Solver        krylov_solver;
  HYPRE_Solver        mgr_precond=NULL, mgr_precond_gotten;

  HYPRE_Int           num_procs, myid;
  HYPRE_Int           time_index;
  MPI_Comm            comm = hypre_MPI_COMM_WORLD;
  HYPRE_Int first_local_row, last_local_row, local_num_rows;
  HYPRE_Int first_local_col, last_local_col, local_num_cols;
  HYPRE_Real *values;

  /* parameters for GMRES */
  HYPRE_Int     k_dim = 100;
  HYPRE_Int     max_iter = 100;
  HYPRE_Int     num_iterations;
  HYPRE_Real    tol = 1.e-6;
  /* mgr options */
  HYPRE_Int mgr_non_c_to_f = 1;

  /* array for constructing the hierarchy */
  HYPRE_Int mgr_bsize = 2;
  HYPRE_Int mgr_nlevels = 1;
  HYPRE_Int *mgr_idx_array = NULL;
  HYPRE_Int *mgr_num_cindexes = NULL;
  HYPRE_Int **mgr_cindexes = NULL;

  /* options for solvers at each level */
  HYPRE_Int mgr_relax_type = 0;
  HYPRE_Int mgr_num_relax_sweeps = 1;

  HYPRE_Int mgr_restrict_type = 0;
  HYPRE_Int mgr_interp_type = 2;

  HYPRE_Int mgr_frelax_method = 99;

  mgr_cindexes = hypre_CTAlloc(HYPRE_Int*, mgr_nlevels, HYPRE_MEMORY_HOST);
  HYPRE_Int *lv1 = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
  lv1[0] = 1;
  mgr_cindexes[0] = lv1;

  mgr_num_cindexes = hypre_CTAlloc(HYPRE_Int, mgr_nlevels, HYPRE_MEMORY_HOST);
  mgr_num_cindexes[0] = 1;

  /* end mgr options */

  /*-----------------------------------------------------------
   * Initialize some stuff
   *-----------------------------------------------------------*/

  /* Initialize MPI */
  hypre_MPI_Init(&argc, &argv);

  hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

  /*-----------------------------------------------------------
   * Set defaults
   *-----------------------------------------------------------*/

  build_matrix_arg_index = argc;
  build_precond_type = 0;
  build_precond_arg_index = argc;
  build_rhs_type = 2;
  build_rhs_arg_index = argc;


  /*-----------------------------------------------------------
   * Parse command line
   *-----------------------------------------------------------*/

  arg_index = 1;

  while (arg_index < argc)
  {
    if ( strcmp(argv[arg_index], "-fromfile") == 0 )
    {
      arg_index++;
      build_matrix_arg_index = arg_index;
    }
    if ( strcmp(argv[arg_index], "-precondfromfile") == 0 )
    {
      arg_index++;
      build_precond_type      = -1;
      build_precond_arg_index = arg_index;
    }
    else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
    {
      arg_index++;
      build_rhs_type      = 0;
      build_rhs_arg_index = arg_index;
    }
    else if ( strcmp(argv[arg_index], "-blockCF") == 0)
    {
      arg_index++;
      build_block_cf_arg_index = arg_index;
    }
    else
    {
      arg_index++;
    }
  }

  if (myid == 0)
  {
    hypre_printf("Reading the system matrix\n");
  }
  ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                    HYPRE_PARCSR, &ij_A );
  if (ierr)
  {
    hypre_printf("ERROR: Problem reading in the system matrix!\n");
    exit(1);
  }

  // Get the CSR matrix from IJ matrix
  ierr = HYPRE_IJMatrixGetLocalRange( ij_A,
                          &first_local_row, &last_local_row ,
                          &first_local_col, &last_local_col );

  local_num_rows = last_local_row - first_local_row + 1;
  local_num_cols = last_local_col - first_local_col + 1;
  ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
  parcsr_A = (HYPRE_ParCSRMatrix) object;


  if (build_precond_type < 0)
  {
    if (myid == 0)
    {
      hypre_printf("Reading the preconditioning matrix\n");
    }
    ierr = HYPRE_IJMatrixRead( argv[build_precond_arg_index], comm,
                      HYPRE_PARCSR, &ij_M );
    if (ierr)
    {
      hypre_printf("ERROR: Problem reading in the preconditioning matrix!\n");
      exit(1);
    }
    ierr = HYPRE_IJMatrixGetLocalRange( ij_M,
                            &first_local_row, &last_local_row ,
                            &first_local_col, &last_local_col );

    local_num_rows = last_local_row - first_local_row + 1;
    local_num_cols = last_local_col - first_local_col + 1;
    ierr += HYPRE_IJMatrixGetObject( ij_M, &object);
    parcsr_M = (HYPRE_ParCSRMatrix) object;
  }
  else
  {
    parcsr_M = parcsr_A;
  }

  if ( build_rhs_type == 0 )
  {
    if (myid == 0)
    {
      hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
      hypre_printf("  Initial guess is 0\n");
    }

    /* RHS */
    ierr = HYPRE_IJVectorRead( argv[build_rhs_arg_index], hypre_MPI_COMM_WORLD,
                      HYPRE_PARCSR, &ij_b );
    if (ierr)
    {
      hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
      exit(1);
    }
    ierr = HYPRE_IJVectorGetObject( ij_b, &object );
    b = (HYPRE_ParVector) object;

    /* Initial guess */
    HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x);

    values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
    hypre_TFree(values, HYPRE_MEMORY_HOST);

    ierr = HYPRE_IJVectorGetObject( ij_x, &object );
    x = (HYPRE_ParVector) object;
  }
  else if ( build_rhs_type == 2 )
  {
    if (myid == 0)
    {
      hypre_printf("  RHS vector has unit components\n");
      hypre_printf("  Initial guess is 0\n");
    }

    /* RHS */
    HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
    HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_b);

    values = hypre_CTAlloc(HYPRE_Real,  local_num_rows, HYPRE_MEMORY_HOST);
    for (i = 0; i < local_num_rows; i++)
      values[i] = 1.0;
    HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
    hypre_TFree(values, HYPRE_MEMORY_HOST);

    ierr = HYPRE_IJVectorGetObject( ij_b, &object );
    b = (HYPRE_ParVector) object;

    /* Initial guess */
    HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x);

    values = hypre_CTAlloc(HYPRE_Real,  local_num_cols, HYPRE_MEMORY_HOST);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
    hypre_TFree(values, HYPRE_MEMORY_HOST);

    ierr = HYPRE_IJVectorGetObject( ij_x, &object );
    x = (HYPRE_ParVector) object;
  }

  // Get the block CF marker for U and P
  mgr_idx_array = hypre_CTAlloc(HYPRE_Int, mgr_bsize, HYPRE_MEMORY_HOST);
  FILE *ifp;
  char fname[80];
  hypre_sprintf(fname, "%s.%05i", argv[build_block_cf_arg_index],myid);
  ifp = fopen(fname,"r");
  if (ifp == NULL) {
    fprintf(stderr, "Can't open input file for block CF indices!\n");
    exit(1);
  }
  for (i = 0; i < mgr_bsize; i++)
  {
    fscanf(ifp, "%d", &mgr_idx_array[i]);
  }

	// setup A_uu block
	hypre_ParCSRMatrix *A_uu = NULL;
	HYPRE_Int nloc = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(parcsr_M));
	HYPRE_Int *CF_marker = hypre_CTAlloc(HYPRE_Int, nloc, HYPRE_MEMORY_HOST);
	for (i = 0; i < nloc; i++)
	{
		if (i < (mgr_idx_array[1] - mgr_idx_array[0]))
		{
			CF_marker[i] = -1;
		}
		else
		{
			CF_marker[i] = 1;
		}
	}
	HYPRE_MGRBuildAffNew(parcsr_M, CF_marker, 0, &A_uu);
	hypre_TFree(CF_marker, HYPRE_MEMORY_HOST);

  // Construct MGR preconditioner
  time_index = hypre_InitializeTiming("GMRES Setup");
  hypre_BeginTiming(time_index);

  HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &krylov_solver);
  HYPRE_GMRESSetKDim(krylov_solver, k_dim);
  HYPRE_GMRESSetMaxIter(krylov_solver, max_iter);
  HYPRE_GMRESSetTol(krylov_solver, tol);
  HYPRE_GMRESSetAbsoluteTol(krylov_solver, tol);
  HYPRE_GMRESSetLogging(krylov_solver, 1);
  HYPRE_GMRESSetPrintLevel(krylov_solver, 2);

  /* use MGR preconditioning */
  if (myid == 0) hypre_printf("Solver:  MGR-GMRES\n");

  HYPRE_MGRCreate(&mgr_precond);

  /* set MGR data by block */
  HYPRE_MGRSetCpointsByContiguousBlock( mgr_precond, mgr_bsize, mgr_nlevels, mgr_idx_array, mgr_num_cindexes, mgr_cindexes);

  /* set intermediate coarse grid strategy */
  HYPRE_MGRSetNonCpointsToFpoints(mgr_precond, mgr_non_c_to_f);
  /* set F relaxation strategy */
  HYPRE_MGRSetFRelaxMethod(mgr_precond, mgr_frelax_method);
  /* set relax type for single level F-relaxation and post-relaxation */
  HYPRE_MGRSetRelaxType(mgr_precond, mgr_relax_type);
  HYPRE_MGRSetNumRelaxSweeps(mgr_precond, mgr_num_relax_sweeps);
  /* set restrict type */
  HYPRE_MGRSetRestrictType(mgr_precond, mgr_restrict_type);
  /* set interpolation type */
  HYPRE_MGRSetInterpType(mgr_precond, mgr_interp_type);
  /* do only 1 iteration for preconditioning */
  HYPRE_MGRSetMaxIter(mgr_precond, 1);
  /* skip global smoother */
  HYPRE_MGRSetMaxGlobalsmoothIters( mgr_precond, 0);

  /* create AMG coarse grid solver */
  HYPRE_BoomerAMGCreate(&mgr_coarse_solver);
  HYPRE_BoomerAMGSetPrintLevel(mgr_coarse_solver, 1);
  HYPRE_BoomerAMGSetRelaxOrder(mgr_coarse_solver, 1);
  HYPRE_BoomerAMGSetMaxIter(mgr_coarse_solver, 1);
  HYPRE_BoomerAMGSetNumSweeps(mgr_coarse_solver, 1);

  /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
  HYPRE_MGRSetCoarseSolver( mgr_precond, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, mgr_coarse_solver);

	/* create the F-relaxation solver for A_uu block */
	HYPRE_Solver aff_solver;
	HYPRE_BoomerAMGCreate(&aff_solver);
  HYPRE_BoomerAMGSetPrintLevel(aff_solver, 1);
	HYPRE_BoomerAMGSetRelaxOrder(aff_solver, 1);
	HYPRE_BoomerAMGSetMaxIter(aff_solver, 1);
	HYPRE_BoomerAMGSetNumFunctions(aff_solver, 3);
	HYPRE_BoomerAMGSetAggNumLevels(aff_solver, 1);

	// setup
	HYPRE_BoomerAMGSetup(aff_solver, A_uu, NULL, NULL);

  // set fine grid solver
  HYPRE_MGRSetFSolver(mgr_precond, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, aff_solver);

  /* setup MGR-PCG solver */
  HYPRE_GMRESSetMaxIter(krylov_solver, max_iter);
  HYPRE_GMRESSetPrecond(krylov_solver,
      (HYPRE_PtrToSolverFcn) HYPRE_MGRSolve,
      (HYPRE_PtrToSolverFcn) HYPRE_MGRSetup,
                    mgr_precond);


  HYPRE_GMRESGetPrecond(krylov_solver, &mgr_precond_gotten);
  if (mgr_precond_gotten != mgr_precond)
  {
    hypre_printf("HYPRE_GMRESGetPrecond got bad precond\n");
    return(-1);
  }
  else
    if (myid == 0)
      hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");


  HYPRE_GMRESSetup
    (krylov_solver, (HYPRE_Matrix)parcsr_M, (HYPRE_Vector)b, (HYPRE_Vector)x);

  hypre_EndTiming(time_index);
  hypre_PrintTiming("Setup phase time", hypre_MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();

  time_index = hypre_InitializeTiming("GMRES Solve");
  hypre_BeginTiming(time_index);

  hypre_ParVectorSetConstantValues(x, 0.0);
  HYPRE_GMRESSolve
    (krylov_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

  hypre_EndTiming(time_index);
  hypre_PrintTiming("Solve phase time", hypre_MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();

  HYPRE_IJVectorPrint(ij_x, "x.out");

  HYPRE_GMRESGetNumIterations(krylov_solver, &num_iterations);
  HYPRE_GMRESGetFinalRelativeResidualNorm(krylov_solver,&final_res_norm);

  // free memory for flex GMRES
  HYPRE_ParCSRGMRESDestroy(krylov_solver);

  HYPRE_MGRDestroy(mgr_precond);
  HYPRE_BoomerAMGDestroy(aff_solver);
  HYPRE_BoomerAMGDestroy(mgr_coarse_solver);

  // Print out solver summary
  if (myid == 0)
  {
    hypre_printf("\n");
    hypre_printf("GMRES Iterations = %d\n", num_iterations);
    hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
    hypre_printf("\n");
  }

  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
  // free the matrix, the rhs and the initial guess
  HYPRE_IJMatrixDestroy(ij_A);
  HYPRE_IJMatrixDestroy(ij_M);
  HYPRE_IJVectorDestroy(ij_b);
  HYPRE_IJVectorDestroy(ij_x);

  hypre_TFree(mgr_idx_array, HYPRE_MEMORY_HOST);
  hypre_TFree(mgr_cindexes, HYPRE_MEMORY_HOST);
  hypre_TFree(mgr_num_cindexes, HYPRE_MEMORY_HOST);
  hypre_TFree(lv1, HYPRE_MEMORY_HOST);

  hypre_MPI_Finalize();

  return (0);
}
