/*
 * Copyright (c) 2009-2015, Peter Abeles. All Rights Reserved.
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

package jmatbench.lattices;

import io.jenetics.lattices.matrix.DoubleMatrix2d;
import io.jenetics.lattices.matrix.blas.Algebra;
import io.jenetics.lattices.matrix.blas.Cholesky;
import io.jenetics.lattices.matrix.blas.Eigenvalue;
import io.jenetics.lattices.matrix.blas.SingularValue;
import jmbench.interfaces.BenchmarkMatrix;
import jmbench.interfaces.DetectedException;
import jmbench.interfaces.MatrixProcessorInterface;
import jmbench.interfaces.RuntimePerformanceFactory;
import jmbench.matrix.RowMajorBenchmarkMatrix;
import jmbench.matrix.RowMajorMatrix;
import jmbench.matrix.RowMajorOps;
import jmbench.tools.BenchmarkConstants;


/**
 * @author Peter Abeles
 */
public class LatticesAlgorithmFactory implements RuntimePerformanceFactory {

    @Override
    public BenchmarkMatrix create(int numRows, int numCols) {
        var mat = DoubleMatrix2d.DENSE.create(numRows, numCols);

        return wrap(mat);
    }

    @Override
    public BenchmarkMatrix wrap(Object matrix) {
        return new LatticesBenchmarkMatrix((DoubleMatrix2d)matrix);
    }

    @Override
    public MatrixProcessorInterface chol() {
        return new Chol();
    }

    public static class Chol implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d L = null;
            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                var chol = Cholesky.decompose(matA);

                if( !chol.isSymmetricPositiveDefinite() ) {
                    throw new DetectedException("Is not SPD");
                }

                L = chol.L();
            }

            long elapsed = System.nanoTime()-prev;
            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(L);
            }
            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface lu() {
        return new LU();
    }

    public static class LU implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d tmp = DoubleMatrix2d.DENSE.create(matA.rows(),matA.cols());

            DoubleMatrix2d L = null;
            DoubleMatrix2d U = null;
            int pivot[] = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                tmp.assign(matA);
                var lu = io.jenetics.lattices.matrix.blas.LU.decompose(tmp);

                L = lu.L();
                U = lu.U();
                pivot = lu.pivot();

                if(!lu.isNonSingular())
                    throw new DetectedException("Singular matrix");
            }

            long elapsed = System.nanoTime()-prev;
            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(L);
                outputs[1] = new LatticesBenchmarkMatrix(U);
                outputs[2] = new RowMajorBenchmarkMatrix(RowMajorOps.pivotMatrix(null, pivot, pivot.length, false));
            }
            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface svd() {
        return new SVD();
    }

    public static class SVD implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d U = null;
            DoubleMatrix2d S = null;
            DoubleMatrix2d V = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                var s = SingularValue.decompose(matA);
                U = s.U();
                S = s.S();
                V = s.V();
            }

            long elapsed = System.nanoTime()-prev;
            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(U);
                outputs[1] = new LatticesBenchmarkMatrix(S);
                outputs[2] = new LatticesBenchmarkMatrix(V);
            }
            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface eigSymm() {
        return new Eig();
    }

    public static class Eig implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d D = null;
            DoubleMatrix2d V = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                var eig = Eigenvalue.decompose(matA);

                D = eig.D();
                V = eig.V();
            }

            long elapsed = System.nanoTime()-prev;
            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(D);
                outputs[1] = new LatticesBenchmarkMatrix(V);
            }
            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface qr() {
        return new QR();
    }

    public static class QR implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d Q = null;
            DoubleMatrix2d R = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                var qr = io.jenetics.lattices.matrix.blas.QR.decompose(matA);

                Q = qr.Q();
                R = qr.R();
            }

            long elapsed = System.nanoTime()-prev;
            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(Q);
                outputs[1] = new LatticesBenchmarkMatrix(R);
            }
            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface det() {
        return new Det();
    }

    public static class Det implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                Algebra.det(matA);
            }

            return System.nanoTime()-prev;
        }
    }

    @Override
    public MatrixProcessorInterface invert() {
        return new Inv();
    }

    public static class Inv implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d result = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                result = Algebra.inverse(matA);
            }

            long elapsed = System.nanoTime()-prev;

            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }

            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface invertSymmPosDef() {
        return new InvSymmPosDef();
    }

    public static class InvSymmPosDef implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d result = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                var chol = Cholesky.decompose(matA);
                var identity = DoubleMatrix2d.DENSE.create(matA.rows(), matA.rows());
                for (int r = 0; r < identity.rows(); ++r) {
                    identity.set(r, r, 1.0);
                }

                result = chol.solve(identity);
            }

            long elapsed = System.nanoTime()-prev;
            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }
            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface add() {
        return new Add();
    }

    public static class Add implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();
            DoubleMatrix2d matB = inputs[1].getOriginal();

            DoubleMatrix2d result = DoubleMatrix2d.DENSE.create(matA.rows(), matA.cols());

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                // In-place operation here
                result.assign(matA);
                result.assign(matB, Double::sum);
            }

            long elapsed = System.nanoTime()-prev;

            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }

            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface mult() {
        return new Mult();
    }

    public static class Mult implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();
            DoubleMatrix2d matB = inputs[1].getOriginal();

            DoubleMatrix2d result = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                result = matA.mult(matB, null);
            }

            long elapsed = System.nanoTime()-prev;

            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }

            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface multTransB() {
        return new MulTranB();
    }

    public static class MulTranB implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();
            DoubleMatrix2d matB = inputs[1].getOriginal();

            DoubleMatrix2d result = DoubleMatrix2d.DENSE.create(matA.cols(), matB.cols());
            
            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                matA.mult(matB, result, 1, 0, false, true);
            }

            long elapsed = System.nanoTime()-prev;

            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }

            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface scale() {
        return new Scale();
    }

    public static class Scale implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();

            DoubleMatrix2d result = DoubleMatrix2d.DENSE.create(matA.rows(), matA.cols());

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                // in-place operator
                result.assign(matA);
                result.assign(a -> a*BenchmarkConstants.SCALE);
            }

            long elapsed = System.nanoTime()-prev;

            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }

            return elapsed;
        }
    }

    @Override
    public MatrixProcessorInterface solveExact() {
        return new Solve();
    }

    @Override
    public MatrixProcessorInterface solveOver() {
        return new Solve();
    }

    public static class Solve implements MatrixProcessorInterface {
        @Override
        public long process(BenchmarkMatrix[] inputs, BenchmarkMatrix[] outputs, long numTrials) {
            DoubleMatrix2d matA = inputs[0].getOriginal();
            DoubleMatrix2d matB = inputs[1].getOriginal();

            DoubleMatrix2d result = null;

            long prev = System.nanoTime();

            for( long i = 0; i < numTrials; i++ ) {
                result = Algebra.solve(matA, matB);
            }

            if( outputs != null ) {
                outputs[0] = new LatticesBenchmarkMatrix(result);
            }

            return System.nanoTime()-prev;
        }
    }

    @Override
    public MatrixProcessorInterface transpose() {
        // yep this is one of "those" libraries that just flags the matrix as being transposed
        return null;
    }

    @Override
    public BenchmarkMatrix convertToLib(RowMajorMatrix input) {
        return new LatticesBenchmarkMatrix(convertToColt(input));
    }

    @Override
    public RowMajorMatrix convertToRowMajor(BenchmarkMatrix input) {
        DoubleMatrix2d mat = input.getOriginal();
        return coltToEjml(mat);
    }

    @Override
    public String getLibraryVersion() {
        return "1.2";
    }

    @Override
    public boolean isNative() {
        return false;
    }

    @Override
    public String getSourceHash() {
        return "";
    }

    public static DoubleMatrix2d convertToColt( RowMajorMatrix orig ) {
        DoubleMatrix2d mat = DoubleMatrix2d.DENSE.create(orig.numRows, orig.numCols);

        for( int i = 0; i < orig.numRows; i++ ) {
            for( int j = 0; j < orig.numCols; j++ ) {
                mat.set(i,j,orig.get(i,j));
            }
        }

        return mat;
    }

    public static RowMajorMatrix coltToEjml( DoubleMatrix2d orig )
    {
        if( orig == null )
            return null;

        RowMajorMatrix mat = new RowMajorMatrix(orig.rows(), orig.cols());

        for( int i = 0; i < mat.numRows; i++ ) {
            for( int j = 0; j < mat.numCols; j++ ) {
                mat.set(i,j,orig.get(i,j));
            }
        }

        return mat;
    }
}