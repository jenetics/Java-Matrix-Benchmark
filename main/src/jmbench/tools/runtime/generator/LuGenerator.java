/*
 * Copyright (c) 2009-2011, Peter Abeles. All Rights Reserved.
 *
 * This file is part of JMatrixBenchmark.
 *
 * JMatrixBenchmark is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * JMatrixBenchmark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JMatrixBenchmark.  If not, see <http://www.gnu.org/licenses/>.
 */

package jmbench.tools.runtime.generator;

import jmbench.interfaces.BenchmarkMatrix;
import jmbench.interfaces.MatrixFactory;
import jmbench.matrix.RowMajorMatrix;
import jmbench.matrix.RowMajorOps;
import jmbench.tools.OutputError;
import jmbench.tools.runtime.InputOutputGenerator;
import jmbench.tools.stability.StabilityBenchmark;

import java.util.Random;

import static jmbench.misc.RandomizeMatrices.randomize;


/**
 * @author Peter Abeles
 */
public class LuGenerator implements InputOutputGenerator {

    RowMajorMatrix A;

    @Override
    public BenchmarkMatrix[] createInputs( MatrixFactory factory , Random rand ,
                                           boolean checkResults , int size ) {
        BenchmarkMatrix[] inputs = new  BenchmarkMatrix[1];

        inputs[0] = factory.create(size,size);

        randomize(inputs[0],-1,1,rand);

        if( checkResults ) {
            A = new RowMajorMatrix(inputs[0]);
        }

        return inputs;
    }

    @Override
    public OutputError checkResults(BenchmarkMatrix[] output, double tol) {

        if( output[0] == null || output[1] == null ) {
            return OutputError.MISC;
        }

        RowMajorMatrix L = new RowMajorMatrix(output[0]);
        RowMajorMatrix U = new RowMajorMatrix(output[1]);
        RowMajorMatrix P = output[2] != null ? new RowMajorMatrix(output[2]) : null;

        if(RowMajorOps.hasUncountable(L) || RowMajorOps.hasUncountable(U) ) {
            return OutputError.UNCOUNTABLE;
        } else if( P != null && RowMajorOps.hasUncountable(P) )
            return OutputError.UNCOUNTABLE;

        if( P == null ) {
            RowMajorMatrix found = RowMajorOps.mult(L,U,null);
            double error = StabilityBenchmark.residualError(found,A);
            if( error > tol ) {
                return OutputError.LARGE_ERROR;
            }
        } else {
            RowMajorMatrix P_tran = RowMajorOps.transpose(P,null);
            RowMajorMatrix PL = RowMajorOps.mult(P_tran,L,null);
            RowMajorMatrix found = RowMajorOps.mult(PL,U,null);

            double error = StabilityBenchmark.residualError(found,A);
            if( error > tol ) {
                return OutputError.LARGE_ERROR;
            }
        }

        return OutputError.NO_ERROR;
    }

    @Override
    public int numOutputs() {
        return 3;
    }

    @Override
    public long getRequiredMemory( int matrixSize ) {
        return 8L*matrixSize*matrixSize*8L;
    }
}