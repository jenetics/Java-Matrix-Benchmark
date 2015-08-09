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
import jmbench.misc.RandomizeMatrices;
import jmbench.tools.OutputError;
import jmbench.tools.runtime.InputOutputGenerator;
import jmbench.tools.stability.StabilityBenchmark;

import java.util.Random;


/**
 * @author Peter Abeles
 */
public class EigSymmGenerator implements InputOutputGenerator {

    RowMajorMatrix A;

    @Override
    public BenchmarkMatrix[] createInputs( MatrixFactory factory , Random rand ,
                                           boolean checkResults , int size ) {
        BenchmarkMatrix[] inputs = new  BenchmarkMatrix[1];

        inputs[0] = factory.create(size,size);

        RandomizeMatrices.symmetric(inputs[0],-1,1,rand);

        if( checkResults ) {
            this.A = new RowMajorMatrix(inputs[0]);
        }

        return inputs;
    }

    @Override
    public OutputError checkResults(BenchmarkMatrix[] output, double tol) {
        if( output[0] == null || output[1] == null ) {
            return OutputError.MISC;
        }

        RowMajorMatrix D = new RowMajorMatrix(output[0]);
        RowMajorMatrix V = new RowMajorMatrix(output[1]);

        RowMajorMatrix L = RowMajorOps.mult(A, V, null);
        RowMajorMatrix R =RowMajorOps.mult(V, D, null);

        if( RowMajorOps.hasUncountable(L) || RowMajorOps.hasUncountable(R) )
            return OutputError.UNCOUNTABLE;

        double error = StabilityBenchmark.residualError(L,R);
        if( error > tol ) {
            return OutputError.LARGE_ERROR;
        }

        return OutputError.NO_ERROR;
    }

    @Override
    public int numOutputs() {
        return 2;
    }

    @Override
    public long getRequiredMemory( int matrixSize ) {
        return 8L*matrixSize*matrixSize*8L;
    }
}